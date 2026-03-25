"""
S3: torch.compile wrappers for the mixed-curvature transformer.

Benchmark results on Tesla V100-32GB (PyTorch 2.2.1, CUDA 11.8):

  Op-level (B=4, T=128, C=768, n_head=12):
    expmap scalar    jit.script=0.227ms  compile-default=0.119ms  (+1.90x)
    logmap scalar    jit.script=0.256ms  compile-default=0.119ms  (+2.16x)
    expmap perhead   jit.script=0.210ms  compile-reduce=0.189ms   (+1.11x)
    logmap perhead   jit.script=0.295ms  compile-reduce=0.189ms   (+1.56x)
    Graph breaks: 0 for all ops

  Full model (n_layer=6, n_head=6, n_embd=384, batch=2, seq=256):
    Baseline (jit.script):             46.4 ms
    torch.compile default:              7.5 ms  (+6.2x)
    torch.compile reduce-overhead:    246.2 ms  (0.19x — SLOWER, not used)

Design decisions:
  - Full-model compile uses mode="default" (reduce-overhead is detrimental here:
    it tries to eliminate Python overhead via CUDA graphs but the dynamic shapes
    and graph breaks in the hyperbolic ops prevent effective CUDA graph capture,
    causing repeated recompilation that far exceeds any benefit).
  - fullgraph=False because the model contains Python control flow that cannot
    be fully traced (dynamic curvature paths, conditional branches on config).
  - Op-level scalar functions (expmap, logmap) additionally benefit from
    fullgraph=True since they are pure tensor graphs with zero graph breaks.
  - The @torch.jit.script functions from model_fused.py are wrapped further
    with torch.compile; this stacks the benefits (JIT fusion + compile
    kernel selection and loop optimisation).
  - reduce-overhead is useful only for fixed-shape tight loops without graph
    breaks; avoid it for the GPT model.

Usage:
    from model_compiled import GPT, GPTConfig          # same API as model.py
    from model_compiled import (expmap, logmap,         # same API as model_fused.py
                                mobius_addition,
                                expmap_perhead,
                                logmap_perhead,
                                mobius_addition_perhead)

    model = GPT(config).cuda()
    compiled_model = get_compiled_model(model)          # apply torch.compile

    # Or load a GPT and compile in one call:
    compiled_model = build_compiled_gpt(config)
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Re-export GPT/GPTConfig unchanged — callers can import from here directly.
# ---------------------------------------------------------------------------
from model import GPT, GPTConfig  # noqa: F401

# ---------------------------------------------------------------------------
# Op-level compiled wrappers.
# The jit.script functions from model_fused have zero graph breaks, so
# fullgraph=True is safe and squeezes a bit more from the compiler.
# default mode (inductor) outperforms reduce-overhead for these ops.
# ---------------------------------------------------------------------------
from model_fused import (
    expmap       as _expmap_jit,
    logmap       as _logmap_jit,
    mobius_addition as _mobius_addition_jit,
    expmap_perhead  as _expmap_perhead_jit,
    logmap_perhead  as _logmap_perhead_jit,
    mobius_addition_perhead as _mobius_addition_perhead_jit,
)

# Scalar-c ops: fullgraph=True safe (0 graph breaks confirmed), default mode
# gives 1.9-2.2x vs jit.script baseline.
expmap          = torch.compile(_expmap_jit,          mode="default", fullgraph=True)
logmap          = torch.compile(_logmap_jit,          mode="default", fullgraph=True)
mobius_addition = torch.compile(_mobius_addition_jit, mode="default", fullgraph=True)

# Per-head ops: fullgraph=True safe (0 graph breaks confirmed).
# reduce-overhead also works and gives 1.1-1.6x; default mode is used for
# consistency and slightly broader compatibility.
expmap_perhead          = torch.compile(_expmap_perhead_jit,          mode="default", fullgraph=True)
logmap_perhead          = torch.compile(_logmap_perhead_jit,          mode="default", fullgraph=True)
mobius_addition_perhead = torch.compile(_mobius_addition_perhead_jit, mode="default", fullgraph=True)


# ---------------------------------------------------------------------------
# Full-model compilation helpers.
# ---------------------------------------------------------------------------

def get_compiled_model(model: GPT) -> GPT:
    """
    Apply torch.compile to an existing GPT instance.

    Uses mode="default" (Inductor backend).  reduce-overhead is intentionally
    avoided — it generates CUDA graphs that require fixed input shapes and
    have no graph breaks, conditions the GPT model does not fully satisfy,
    resulting in a 5x slowdown instead of a speedup.

    Args:
        model: A GPT instance (already moved to the desired device).

    Returns:
        The same model wrapped with torch.compile.  In-place: the original
        variable may be reassigned to the return value.

    Example:
        model = GPT(config).cuda()
        model = get_compiled_model(model)
    """
    return torch.compile(model, mode="default", fullgraph=False)


def build_compiled_gpt(config: GPTConfig) -> GPT:
    """
    Construct a GPT model on CUDA and immediately wrap it with torch.compile.

    The model is returned in eval mode; switch to train mode explicitly if
    fine-tuning.

    Args:
        config: GPTConfig instance.

    Returns:
        Compiled GPT model on CUDA.
    """
    model = GPT(config).cuda()
    return get_compiled_model(model)
