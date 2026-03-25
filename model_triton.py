"""
S5: Custom Triton kernels for the mixed-curvature transformer.

Fuses memory-bandwidth-bound hyperbolic ops (expmap, logmap, mobius_addition)
into single Triton kernels to eliminate intermediate HBM writes.

S1 profiling showed these ops were the primary overhead source (9.36x vs Euclidean).
S3 (torch.compile) brought this to ~1.5x overhead. S5 Triton kernels target <1.3x.

Key insight: each (B,T) token row can be processed independently — one Triton
program per row loads x and v once, computes all intermediates in registers/SRAM,
and writes the result once. This eliminates O(4-6) intermediate HBM round-trips
per op.

The S4 precompute buffers (_c_clamped_cache, _sqrt_c_cache, _inv_sqrt_c_cache)
are exposed as GPU tensors; the triton wrappers accept the float value directly
(already fetched from cache on the Python side).

Fallback: if Triton is unavailable, falls back to the torch.compile-fused ops
from model_fused.py with a warning.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

# ---------------------------------------------------------------------------
# Triton availability check
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    import warnings
    warnings.warn(
        "Triton not available; S5 falling back to torch.compile-fused ops. "
        "Install with: pip install triton",
        RuntimeWarning,
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# Kernel 1: fused_mobius_add
#
# Möbius addition in the Poincaré ball with curvature c:
#   x_ns = ||x||^2
#   y_ns = ||y||^2
#   ip   = <x, y>
#   num  = (1 + 2c*ip + c*y_ns)*x + (1 - c*x_ns)*y
#   den  = 1 + 2c*ip + c^2*x_ns*y_ns + eps
#   out  = num / den
#
# One Triton program handles one (B,T) row of C elements.
# Two passes over the row: first to compute the scalar reductions,
# then to compute and store the output (avoids storing intermediate vectors).
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:
    @triton.jit
    def _mobius_add_kernel(
        x_ptr, y_ptr, out_ptr,
        C,           # embedding dim (runtime value)
        c_val,       # curvature as float32 scalar
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        base = row * C
        offs = tl.arange(0, BLOCK_C)

        # --- Pass 1: accumulate scalar reductions ---
        x_ns = tl.zeros([1], dtype=tl.float32)
        y_ns = tl.zeros([1], dtype=tl.float32)
        ip   = tl.zeros([1], dtype=tl.float32)

        for k in range(0, C, BLOCK_C):
            mask = (offs + k) < C
            xv = tl.load(x_ptr + base + offs + k, mask=mask, other=0.0).to(tl.float32)
            yv = tl.load(y_ptr + base + offs + k, mask=mask, other=0.0).to(tl.float32)
            x_ns += tl.sum(xv * xv, axis=0)
            y_ns += tl.sum(yv * yv, axis=0)
            ip   += tl.sum(xv * yv, axis=0)

        # Scalar coefficients (shape [1])
        denom   = 1.0 + 2.0 * c_val * ip + c_val * c_val * x_ns * y_ns + 1e-9
        coeff_x = (1.0 + 2.0 * c_val * ip + c_val * y_ns) / denom
        coeff_y = (1.0 - c_val * x_ns) / denom

        # --- Pass 2: write fused output ---
        for k in range(0, C, BLOCK_C):
            mask = (offs + k) < C
            xv = tl.load(x_ptr + base + offs + k, mask=mask, other=0.0).to(tl.float32)
            yv = tl.load(y_ptr + base + offs + k, mask=mask, other=0.0).to(tl.float32)
            out = coeff_x * xv + coeff_y * yv
            tl.store(out_ptr + base + offs + k, out, mask=mask)


    @triton.jit
    def _expmap_kernel(
        x_ptr, v_ptr, out_ptr,
        C,
        c_val,
        BLOCK_C: tl.constexpr,
    ):
        """
        Fused expmap: maps tangent vector v at base point x to the Poincaré ball.

        Formula:
          c_cl  = clamp(c, 1e-4, 1.0)   -- handled by caller passing clamped value
          sf_x  = 2 / (1 + c*||x||^2 + eps)          # conformal factor at x
          v_n   = ||v|| + eps
          sqrt_c = sqrt(c + eps)
          targ  = sqrt(|c * sf_x * ||v||^2 / 2| + eps)
          coeff = tanh(targ) / (sqrt_c + eps)
          second_term = coeff * v / v_n              # move in tangent direction
          result = mobius_add(x, second_term, c)
        """
        row  = tl.program_id(0)
        base = row * C
        offs = tl.arange(0, BLOCK_C)

        # --- Pass 1: norms ---
        x_ns  = tl.zeros([1], dtype=tl.float32)
        v_ns  = tl.zeros([1], dtype=tl.float32)

        for k in range(0, C, BLOCK_C):
            mask = (offs + k) < C
            xv = tl.load(x_ptr + base + offs + k, mask=mask, other=0.0).to(tl.float32)
            vv = tl.load(v_ptr + base + offs + k, mask=mask, other=0.0).to(tl.float32)
            x_ns += tl.sum(xv * xv, axis=0)
            v_ns += tl.sum(vv * vv, axis=0)

        # Scalar intermediates
        eps      = 1e-9
        sf_x     = 2.0 / (1.0 + c_val * x_ns + eps)
        v_norm   = tl.sqrt(v_ns + eps)
        sqrt_c   = tl.sqrt(c_val + eps)

        tanh_arg = tl.sqrt(tl.abs(c_val * sf_x * v_ns / 2.0) + eps)
        # Implement tanh via exp: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        exp2x = tl.exp(2.0 * tanh_arg)
        tanh_val = (exp2x - 1.0) / (exp2x + 1.0)
        coeff_v  = tanh_val / (sqrt_c + eps)  # second_term = coeff_v * v / v_norm

        # --- Reductions needed for Mobius add of x + second_term ---
        # second_term = coeff_v / v_norm * v  =>  ||second_term||^2 = (coeff_v/v_norm)^2 * v_ns
        st_scale = coeff_v / (v_norm + eps)
        st_ns    = st_scale * st_scale * v_ns

        # x_ns already known; need <x, second_term> = st_scale * <x, v>
        xv_ip = tl.zeros([1], dtype=tl.float32)
        for k in range(0, C, BLOCK_C):
            mask = (offs + k) < C
            xv = tl.load(x_ptr + base + offs + k, mask=mask, other=0.0).to(tl.float32)
            vv = tl.load(v_ptr + base + offs + k, mask=mask, other=0.0).to(tl.float32)
            xv_ip += tl.sum(xv * vv, axis=0)
        xst_ip = st_scale * xv_ip  # <x, second_term>

        # Möbius add coefficients
        denom_m   = 1.0 + 2.0 * c_val * xst_ip + c_val * c_val * x_ns * st_ns + eps
        coeff_x_m = (1.0 + 2.0 * c_val * xst_ip + c_val * st_ns) / denom_m
        coeff_st_m = (1.0 - c_val * x_ns) / denom_m

        # --- Pass 3: write output ---
        for k in range(0, C, BLOCK_C):
            mask = (offs + k) < C
            xv = tl.load(x_ptr + base + offs + k, mask=mask, other=0.0).to(tl.float32)
            vv = tl.load(v_ptr + base + offs + k, mask=mask, other=0.0).to(tl.float32)
            st = st_scale * vv  # second_term component
            out = coeff_x_m * xv + coeff_st_m * st
            tl.store(out_ptr + base + offs + k, out, mask=mask)


    @triton.jit
    def _logmap_kernel(
        x_ptr, u_ptr, out_ptr,
        C,
        c_val,
        BLOCK_C: tl.constexpr,
    ):
        """
        Fused logmap: maps point u on the ball to the tangent space at x.

        Formula:
          mob     = mobius_add(-x, u, c)
          sf_x    = 2 / (1 + c*||x||^2 + eps)
          mob_n   = ||mob|| + eps
          sqrt_c  = sqrt(c + eps)
          atanh_a = clamp(sqrt_c * mob_n, -0.9999, 0.9999)
          result  = (2 / (sf_x * sqrt_c + eps)) * arctanh(atanh_a) * mob / mob_n
        """
        row  = tl.program_id(0)
        base = row * C
        offs = tl.arange(0, BLOCK_C)

        eps = 1e-9

        # --- Pass 1: compute scalar reductions for mobius_add(-x, u) ---
        nx_ns = tl.zeros([1], dtype=tl.float32)  # ||-x||^2 = ||x||^2
        u_ns  = tl.zeros([1], dtype=tl.float32)
        nxu_ip = tl.zeros([1], dtype=tl.float32)  # <-x, u>

        for k in range(0, C, BLOCK_C):
            mask = (offs + k) < C
            xv = tl.load(x_ptr + base + offs + k, mask=mask, other=0.0).to(tl.float32)
            uv = tl.load(u_ptr + base + offs + k, mask=mask, other=0.0).to(tl.float32)
            nx_ns  += tl.sum(xv * xv, axis=0)
            u_ns   += tl.sum(uv * uv, axis=0)
            nxu_ip += tl.sum((-xv) * uv, axis=0)

        # Möbius add(-x, u) coefficients
        denom_mob    = 1.0 + 2.0 * c_val * nxu_ip + c_val * c_val * nx_ns * u_ns + eps
        coeff_nx_mob = (1.0 + 2.0 * c_val * nxu_ip + c_val * u_ns) / denom_mob
        coeff_u_mob  = (1.0 - c_val * nx_ns) / denom_mob

        # mob = coeff_nx_mob * (-x) + coeff_u_mob * u
        # ||mob||^2 — need a pass to compute it
        mob_ns = tl.zeros([1], dtype=tl.float32)
        for k in range(0, C, BLOCK_C):
            mask = (offs + k) < C
            xv = tl.load(x_ptr + base + offs + k, mask=mask, other=0.0).to(tl.float32)
            uv = tl.load(u_ptr + base + offs + k, mask=mask, other=0.0).to(tl.float32)
            mob_comp = coeff_nx_mob * (-xv) + coeff_u_mob * uv
            mob_ns += tl.sum(mob_comp * mob_comp, axis=0)

        # Scalar factors for logmap
        x_ns    = nx_ns  # same value
        sf_x    = 2.0 / (1.0 + c_val * x_ns + eps)
        sqrt_c  = tl.sqrt(c_val + eps)
        mob_norm = tl.sqrt(mob_ns + eps)

        atanh_arg = sqrt_c * mob_norm
        # Clamp to (-0.9999, 0.9999)
        atanh_arg = tl.where(atanh_arg >  0.9999,  0.9999, atanh_arg)
        atanh_arg = tl.where(atanh_arg < -0.9999, -0.9999, atanh_arg)

        # arctanh(x) = 0.5 * ln((1+x)/(1-x))
        atanh_val = 0.5 * tl.log((1.0 + atanh_arg) / (1.0 - atanh_arg + eps) + eps)

        # Scale: (2 / (sf_x * sqrt_c + eps)) * atanh_val / mob_norm
        scale = (2.0 / (sf_x * sqrt_c + eps)) * atanh_val / (mob_norm + eps)

        # --- Pass 3: write result = scale * mob ---
        for k in range(0, C, BLOCK_C):
            mask = (offs + k) < C
            xv = tl.load(x_ptr + base + offs + k, mask=mask, other=0.0).to(tl.float32)
            uv = tl.load(u_ptr + base + offs + k, mask=mask, other=0.0).to(tl.float32)
            mob_comp = coeff_nx_mob * (-xv) + coeff_u_mob * uv
            out = scale * mob_comp
            tl.store(out_ptr + base + offs + k, out, mask=mask)


# ---------------------------------------------------------------------------
# Helper: choose BLOCK_C (must be power of 2 for Triton)
# ---------------------------------------------------------------------------

def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _block_c_for(C: int) -> int:
    """Pick a BLOCK_C that covers C in as few tiles as possible, <=256."""
    if C <= 32:
        return max(32, _next_power_of_2(C))
    if C <= 64:
        return 64
    if C <= 128:
        return 128
    return 256   # tile over C in multiple passes if C > 256


# ---------------------------------------------------------------------------
# Public Python wrappers
# ---------------------------------------------------------------------------

def mobius_add_triton(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """
    Triton-fused Möbius addition.

    Args:
        x: (B, T, C) float32 tensor on CUDA in the Poincaré ball
        y: (B, T, C) float32 tensor on CUDA in the Poincaré ball
        c: curvature scalar (Python float, pre-clamped to [1e-4, 1.0])

    Returns:
        out: (B, T, C) float32 result tensor
    """
    if not TRITON_AVAILABLE:
        from model_fused import mobius_addition
        c_t = torch.tensor(c, device=x.device, dtype=x.dtype)
        return mobius_addition(x.flatten(0, 1), y.flatten(0, 1), c_t).view_as(x)

    assert x.is_cuda and y.is_cuda, "Inputs must be on CUDA"
    assert x.shape == y.shape
    assert x.dtype == torch.float32

    x_c = x.contiguous()
    y_c = y.contiguous()
    out = torch.empty_like(x_c)

    B, T, C = x.shape
    n_rows = B * T
    BLOCK_C = _block_c_for(C)

    _mobius_add_kernel[(n_rows,)](
        x_c, y_c, out,
        C=C,
        c_val=float(c),
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


def expmap_triton(x: torch.Tensor, v: torch.Tensor, c: float) -> torch.Tensor:
    """
    Triton-fused exponential map.

    Maps tangent vector v at base point x onto the Poincaré ball.

    Args:
        x: (B, T, C) float32 — base point on the ball
        v: (B, T, C) float32 — tangent vector at x
        c: curvature scalar (Python float, pre-clamped)

    Returns:
        out: (B, T, C) float32
    """
    if not TRITON_AVAILABLE:
        from model_fused import expmap
        c_t = torch.tensor(c, device=x.device, dtype=x.dtype)
        return expmap(x.flatten(0, 1), v.flatten(0, 1), c_t).view_as(x)

    assert x.is_cuda and v.is_cuda
    assert x.shape == v.shape
    assert x.dtype == torch.float32

    x_c = x.contiguous()
    v_c = v.contiguous()
    out = torch.empty_like(x_c)

    B, T, C = x.shape
    n_rows = B * T
    BLOCK_C = _block_c_for(C)

    _expmap_kernel[(n_rows,)](
        x_c, v_c, out,
        C=C,
        c_val=float(c),
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


def logmap_triton(x: torch.Tensor, u: torch.Tensor, c: float) -> torch.Tensor:
    """
    Triton-fused logarithmic map.

    Maps point u on the ball to the tangent space at x.

    Args:
        x: (B, T, C) float32 — base point on the ball
        u: (B, T, C) float32 — point on the ball to map to tangent space
        c: curvature scalar (Python float, pre-clamped)

    Returns:
        out: (B, T, C) float32 tangent vector at x
    """
    if not TRITON_AVAILABLE:
        from model_fused import logmap
        c_t = torch.tensor(c, device=x.device, dtype=x.dtype)
        return logmap(x.flatten(0, 1), u.flatten(0, 1), c_t).view_as(x)

    assert x.is_cuda and u.is_cuda
    assert x.shape == u.shape
    assert x.dtype == torch.float32

    x_c = x.contiguous()
    u_c = u.contiguous()
    out = torch.empty_like(x_c)

    B, T, C = x.shape
    n_rows = B * T
    BLOCK_C = _block_c_for(C)

    _logmap_kernel[(n_rows,)](
        x_c, u_c, out,
        C=C,
        c_val=float(c),
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# Per-head variants (for models with per_head_curvature=True)
#
# Each head slice has dimension hs = C // n_head. We launch B*T*n_head programs,
# each handling one (batch, time, head) slice of size hs.
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:
    @triton.jit
    def _mobius_add_perhead_kernel(
        x_ptr, y_ptr, out_ptr,
        hs,          # head size (runtime)
        c_ptr,       # pointer to (n_head,) curvature tensor
        n_head,      # number of heads (runtime)
        BLOCK_HS: tl.constexpr,
    ):
        """One program per (B, T, head) slice."""
        prog = tl.program_id(0)
        head_idx = prog % n_head
        row      = prog // n_head   # which (B, T) row

        base = row * n_head * hs + head_idx * hs
        offs = tl.arange(0, BLOCK_HS)

        c_val = tl.load(c_ptr + head_idx).to(tl.float32)

        # Pass 1: reductions
        x_ns = tl.zeros([1], dtype=tl.float32)
        y_ns = tl.zeros([1], dtype=tl.float32)
        ip   = tl.zeros([1], dtype=tl.float32)

        for k in range(0, hs, BLOCK_HS):
            mask = (offs + k) < hs
            xv = tl.load(x_ptr + base + offs + k, mask=mask, other=0.0).to(tl.float32)
            yv = tl.load(y_ptr + base + offs + k, mask=mask, other=0.0).to(tl.float32)
            x_ns += tl.sum(xv * xv, axis=0)
            y_ns += tl.sum(yv * yv, axis=0)
            ip   += tl.sum(xv * yv, axis=0)

        denom   = 1.0 + 2.0 * c_val * ip + c_val * c_val * x_ns * y_ns + 1e-9
        coeff_x = (1.0 + 2.0 * c_val * ip + c_val * y_ns) / denom
        coeff_y = (1.0 - c_val * x_ns) / denom

        # Pass 2: output
        for k in range(0, hs, BLOCK_HS):
            mask = (offs + k) < hs
            xv = tl.load(x_ptr + base + offs + k, mask=mask, other=0.0).to(tl.float32)
            yv = tl.load(y_ptr + base + offs + k, mask=mask, other=0.0).to(tl.float32)
            tl.store(out_ptr + base + offs + k, coeff_x * xv + coeff_y * yv, mask=mask)


def mobius_add_perhead_triton(
    x: torch.Tensor,       # (B, T, C)
    y: torch.Tensor,       # (B, T, C)
    c: torch.Tensor,       # (n_head,) curvature per head (float32, clamped)
) -> torch.Tensor:
    """Triton-fused per-head Möbius addition."""
    if not TRITON_AVAILABLE:
        from model_fused import mobius_addition_perhead
        n_head = c.shape[0]
        return mobius_addition_perhead(x, y, c.unsqueeze(0), n_head)

    B, T, C = x.shape
    n_head = c.shape[0]
    hs = C // n_head

    x_c = x.contiguous()
    y_c = y.contiguous()
    c_c = c.contiguous().float()
    out = torch.empty_like(x_c)

    BLOCK_HS = _block_c_for(hs)
    n_rows = B * T
    grid = (n_rows * n_head,)

    _mobius_add_perhead_kernel[grid](
        x_c, y_c, out,
        hs=hs,
        c_ptr=c_c,
        n_head=n_head,
        BLOCK_HS=BLOCK_HS,
        num_warps=4,
    )
    return out
