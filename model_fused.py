"""
Fused hyperbolic operations using torch.jit.script.
The JIT compiler fuses adjacent element-wise ops (mul, add, tanh, sqrt) into
fewer CUDA kernels, reducing the kernel explosion observed in S1 profiling.

Drop-in replacement signatures matching model.py.
Focuses on the common per-head dynamic curvature path that accounts for
most kernel launches.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

# ---------------------------------------------------------------------------
# Core fused ops — @torch.jit.script forces TorchScript which enables
# kernel fusion for adjacent elementwise ops.
# ---------------------------------------------------------------------------

@torch.jit.script
def clamp_curvature(c: Tensor) -> Tensor:
    """Clamp curvature to valid range."""
    return torch.clamp(c, min=1e-4, max=1.0)


@torch.jit.script
def _mobius_add_perhead(
    x_r: Tensor,   # (B, T, n_head, hs)
    y_r: Tensor,   # (B, T, n_head, hs)
    c_r: Tensor,   # (B, 1, n_head, 1) or (1, 1, n_head, 1)
) -> Tensor:
    """Fused Mobius addition in per-head reshaped space."""
    x_norm_sq = torch.sum(x_r * x_r, dim=-1, keepdim=True)
    y_norm_sq = torch.sum(y_r * y_r, dim=-1, keepdim=True)
    inner_product = torch.sum(x_r * y_r, dim=-1, keepdim=True)

    numerator = (1.0 + 2.0 * c_r * inner_product + c_r * y_norm_sq) * x_r + \
                (1.0 - c_r * x_norm_sq) * y_r
    denominator = 1.0 + 2.0 * c_r * inner_product + c_r * c_r * x_norm_sq * y_norm_sq
    return numerator / (denominator + 1e-9)


@torch.jit.script
def _scaling_factor_perhead(
    x_r: Tensor,   # (B, T, n_head, hs)
    c_r: Tensor,   # (B, 1, n_head, 1) or (1, 1, n_head, 1)
) -> Tensor:
    """Fused scaling factor in per-head reshaped space."""
    x_norm_sq = torch.sum(x_r * x_r, dim=-1, keepdim=True)
    return 2.0 / (1.0 + c_r * x_norm_sq + 1e-9)


@torch.jit.script
def expmap_perhead(
    x: Tensor,   # (B, T, C)
    v: Tensor,   # (B, T, C)
    c: Tensor,   # (B, n_head) dynamic curvature
    n_head: int,
) -> Tensor:
    """
    Fused exponential map for dynamic per-head curvature.
    Entire computation compiled as a single TorchScript function,
    allowing nvFuser / NNC to fuse elementwise ops.
    """
    B = x.shape[0]
    T = x.shape[1]
    C_embed = x.shape[2]
    hs = C_embed // n_head

    c_clamped = torch.clamp(c, min=1e-4, max=1.0)           # (B, n_head)
    c_r = c_clamped.view(B, 1, n_head, 1)                   # (B, 1, n_head, 1)

    x_r = x.view(B, T, n_head, hs)
    v_r = v.view(B, T, n_head, hs)

    # Scaling factor
    x_norm_sq_r = torch.sum(x_r * x_r, dim=-1, keepdim=True)
    sf_x_r = 2.0 / (1.0 + c_r * x_norm_sq_r + 1e-9)

    # v norm
    v_norm_sq_r = torch.sum(v_r * v_r, dim=-1, keepdim=True)
    v_norm_r = torch.sqrt(v_norm_sq_r + 1e-9)

    # tanh argument
    tanh_arg_val = torch.abs(c_r * sf_x_r * v_norm_sq_r / 2.0)
    sqrt_c_r = torch.sqrt(torch.abs(c_r) + 1e-9)
    sqrt_tanh_arg = torch.sqrt(tanh_arg_val + 1e-9)

    second_term_coeff = (1.0 / (sqrt_c_r + 1e-9)) * torch.tanh(sqrt_tanh_arg)
    second_term_r = second_term_coeff * (v_r / (v_norm_r + 1e-9))

    # Mobius addition
    result_r = _mobius_add_perhead(x_r, second_term_r, c_r)
    return result_r.reshape(B, T, C_embed)


@torch.jit.script
def logmap_perhead(
    x: Tensor,   # (B, T, C)
    u: Tensor,   # (B, T, C)
    c: Tensor,   # (B, n_head) dynamic curvature
    n_head: int,
) -> Tensor:
    """
    Fused logarithmic map for dynamic per-head curvature.
    """
    B = x.shape[0]
    T = x.shape[1]
    C_embed = x.shape[2]
    hs = C_embed // n_head

    c_clamped = torch.clamp(c, min=1e-4, max=1.0)
    c_r = c_clamped.view(B, 1, n_head, 1)

    x_r = x.view(B, T, n_head, hs)
    u_r = u.view(B, T, n_head, hs)
    neg_x_r = -x_r

    # Mobius addition of -x and u
    mob_r = _mobius_add_perhead(neg_x_r, u_r, c_r)

    # Scaling factor at x
    x_norm_sq_r = torch.sum(x_r * x_r, dim=-1, keepdim=True)
    sf_x_r = 2.0 / (1.0 + c_r * x_norm_sq_r + 1e-9)

    # Norm of mob result
    mob_norm_sq_r = torch.sum(mob_r * mob_r, dim=-1, keepdim=True)
    mob_norm_r = torch.sqrt(mob_norm_sq_r + 1e-9)

    sqrt_c_r = torch.sqrt(torch.abs(c_r) + 1e-9)
    constant_factor = 2.0 / (sf_x_r * sqrt_c_r + 1e-9)
    direction_factor = mob_r / (mob_norm_r + 1e-9)

    arctanh_arg = sqrt_c_r * mob_norm_r
    arctanh_arg_clamped = torch.clamp(arctanh_arg, min=-0.9999, max=0.9999)

    result_r = constant_factor * torch.arctanh(arctanh_arg_clamped) * direction_factor
    return result_r.reshape(B, T, C_embed)


@torch.jit.script
def mobius_addition_perhead(
    x: Tensor,   # (B, T, C)
    y: Tensor,   # (B, T, C)
    c: Tensor,   # (B, n_head) dynamic curvature
    n_head: int,
) -> Tensor:
    """Fused Mobius addition for dynamic per-head curvature."""
    B = x.shape[0]
    T = x.shape[1]
    C_embed = x.shape[2]
    hs = C_embed // n_head

    c_clamped = torch.clamp(c, min=1e-4, max=1.0)
    c_r = c_clamped.view(B, 1, n_head, 1)

    x_r = x.view(B, T, n_head, hs)
    y_r = y.view(B, T, n_head, hs)

    result_r = _mobius_add_perhead(x_r, y_r, c_r)
    return result_r.reshape(B, T, C_embed)


# ---------------------------------------------------------------------------
# Scalar-c fallback (for the non-per-head path, also fused)
# ---------------------------------------------------------------------------

@torch.jit.script
def expmap(x: Tensor, v: Tensor, c: Tensor) -> Tensor:
    """Fused expmap for scalar curvature."""
    c_clamped = torch.clamp(c, min=1e-4, max=1.0)

    x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
    sf_x = 2.0 / (1.0 + c_clamped * x_norm_sq + 1e-9)

    v_norm_sq = torch.sum(v * v, dim=-1, keepdim=True)
    v_norm = torch.sqrt(v_norm_sq + 1e-9)

    sqrt_c = torch.sqrt(c_clamped + 1e-9)
    tanh_arg = torch.sqrt(torch.abs(c_clamped * sf_x * v_norm_sq / 2.0) + 1e-9)
    term_coeff = (1.0 / (sqrt_c + 1e-9)) * torch.tanh(tanh_arg)
    second_term = term_coeff * (v / (v_norm + 1e-9))

    # Inline mobius_addition for fusion
    x_ns = torch.sum(x * x, dim=-1, keepdim=True)
    y_ns = torch.sum(second_term * second_term, dim=-1, keepdim=True)
    ip = torch.sum(x * second_term, dim=-1, keepdim=True)
    num = (1.0 + 2.0 * c_clamped * ip + c_clamped * y_ns) * x + (1.0 - c_clamped * x_ns) * second_term
    den = 1.0 + 2.0 * c_clamped * ip + c_clamped * c_clamped * x_ns * y_ns
    return num / (den + 1e-9)


@torch.jit.script
def logmap(x: Tensor, u: Tensor, c: Tensor) -> Tensor:
    """Fused logmap for scalar curvature."""
    c_clamped = torch.clamp(c, min=1e-4, max=1.0)

    # mobius_addition(-x, u)
    neg_x = -x
    nx_ns = torch.sum(neg_x * neg_x, dim=-1, keepdim=True)
    u_ns = torch.sum(u * u, dim=-1, keepdim=True)
    ip_nu = torch.sum(neg_x * u, dim=-1, keepdim=True)
    mob_num = (1.0 + 2.0 * c_clamped * ip_nu + c_clamped * u_ns) * neg_x + (1.0 - c_clamped * nx_ns) * u
    mob_den = 1.0 + 2.0 * c_clamped * ip_nu + c_clamped * c_clamped * nx_ns * u_ns
    mob = mob_num / (mob_den + 1e-9)

    x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
    sf_x = 2.0 / (1.0 + c_clamped * x_norm_sq + 1e-9)

    mob_norm_sq = torch.sum(mob * mob, dim=-1, keepdim=True)
    mob_norm = torch.sqrt(mob_norm_sq + 1e-9)

    sqrt_c = torch.sqrt(c_clamped + 1e-9)
    constant_factor = 2.0 / (sf_x * sqrt_c + 1e-9)
    direction = mob / (mob_norm + 1e-9)
    arctanh_arg = torch.clamp(sqrt_c * mob_norm, min=-0.9999, max=0.9999)
    return constant_factor * torch.arctanh(arctanh_arg) * direction


@torch.jit.script
def mobius_addition(x: Tensor, y: Tensor, c: Tensor) -> Tensor:
    """Fused mobius addition for scalar curvature."""
    c_clamped = torch.clamp(c, min=1e-4, max=1.0)
    x_ns = torch.sum(x * x, dim=-1, keepdim=True)
    y_ns = torch.sum(y * y, dim=-1, keepdim=True)
    ip = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1.0 + 2.0 * c_clamped * ip + c_clamped * y_ns) * x + (1.0 - c_clamped * x_ns) * y
    den = 1.0 + 2.0 * c_clamped * ip + c_clamped * c_clamped * x_ns * y_ns
    return num / (den + 1e-9)
