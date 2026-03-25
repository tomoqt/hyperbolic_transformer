# Mixed-Curvature Transformer Validation

## Motivation

This project asks whether introducing hyperbolic structure inside a GPT-style transformer changes representation geometry in a useful way and whether that geometric change can improve optimization. The working hypothesis is that a mixed-curvature residual stream can make hidden states more isotropic than a matched Euclidean baseline and, in the right regime, make training easier rather than harder.

## What Was Actually Validated?

The core validated model is **mixed-curvature**, not fully hyperbolic end to end.

- Embeddings are summed in Euclidean space.
- In the main validation setup, `use_embedding_curvature=False`.
- Transformer blocks apply hyperbolic residual-style updates internally.
- Final norm and LM head remain Euclidean.

The original March 20 validation used:

- `curvature_mode=random`
- `dynamic_curvature=True`
- `per_head_curvature=True`
- `use_embedding_curvature=False`

Later ablations found that this was not the best quality/simplicity tradeoff.

## Validation Results

All main validation runs used `1500` optimizer steps per run.

- Shakespeare: `3072` tokens/step, `4.608M` tokens/run
- FineWeb: `4096` tokens/step, `6.144M` tokens/run

### 1. Representation isotropy is a real effect

Under the original AdamW-style setup:

- Shakespeare isotropy improved mainly in deeper layers.
- FineWeb isotropy improved across all six probed layers.

So the isotropy hypothesis is supported in the original setup, and it becomes cleaner in the larger-data FineWeb regime.

### 2. Pretraining speed improved in the original setup, but only in a specific regime

Under the original optimizer:

- Shakespeare coarse LR sweep: mixed-curvature beat baseline across the full sweep.
- FineWeb coarse LR sweep: mixed-curvature won at `5e-4` and `1e-3`, while baseline won at `1e-4` and `2e-4`.

Interpretation: the optimization-speed signal is real, but it is concentrated in a higher-learning-rate band on FineWeb rather than being universal.

## Muon Extension

The Muon branch asked whether the earlier signal survives a substantially different optimizer.

### Muon isotropy

The isotropy effect survives Muon, but with a less uniform layer profile.

- Shakespeare Muon isotropy: deeper layers improve most strongly.
- FineWeb Muon isotropy: layers `0-1` regress, while layers `2-5` improve progressively.

Interpretation: Muon does not erase the geometry effect, but it shifts where that gain appears.

### Muon speed

The Muon speed branch did **not** reproduce the original acceleration result.

- Shakespeare Muon coarse LR sweep: baseline remained stable across the sweep.
- Mixed-curvature diverged to `NaN` at every swept learning rate under the original Muon regime (`muon_lr_ratio=3`).
- Because the Shakespeare gate already falsified that regime, the downstream FineWeb Muon speed subtree was intentionally canceled.

Interpretation: the speed gain is not yet optimizer-robust.

## Ablation Suite (A0-A7)

The ablation suite isolated which mixed-curvature ingredients actually matter.

Main outcome:

- the best overall quality came from **fixed curvature `c=0.1`**
- dynamic curvature was neutral in this setup
- per-head curvature gave only a tiny gain
- embedding curvature hurt slightly
- Muon remained unstable in the tested ablation regime

Best-quality config from the graph:

```python
curvature_mode = "fixed"
curvature = 0.1
dynamic_curvature = False
per_head_curvature = False
use_embedding_curvature = False
optimizer = "AdamW"
learning_rate = 5e-4
```

Result:

- FineWeb `val_loss = 5.5822`
- Euclidean baseline `val_loss = 6.022`
- improvement: about `-0.44` nats

This matters because the simpler static configuration outperformed the earlier dynamic/per-head validation setup.

## Speed Optimization Branch

The matched FineWeb `5e-4` rerun established a large baseline overhead:

- baseline mean logged step time: `461.88 ms`
- mixed-curvature mean logged step time: `1466.89 ms`
- end-to-end slowdown: about `3.18x`
- train-step-only slowdown: about `5.86x`

The March 25 speed branch then attacked that overhead directly.

### Implemented optimization steps

- `model_fused.py`: TorchScript-fused hyperbolic ops
- `model_compiled.py`: `torch.compile(mode="default", fullgraph=False)` helpers
- `model_precompute.py`: cached static-curvature transforms
- `model_triton.py`: custom Triton kernels for `mobius_addition`, `expmap`, and `logmap`
- `bench_s4.py`, `bench_s5.py`: benchmark harnesses committed in the repo

### Benchmark-backed findings

- TorchScript fusion reduced kernel launches by about `71%` in the profiling branch.
- `torch.compile(..., mode="default")` produced the largest model-level speedup in the benchmark harness.
- Triton kernels improved core primitive throughput:
  - `mobius_addition`: `1.70x`
  - `expmap`: `2.14x`
  - `logmap`: `3.42x`

The graph-level recommendation is to treat that stack as the correct implementation path for closing the mixed-curvature overhead gap.

## Current Recommendations

### Best quality config

Use the ablation winner:

```python
curvature_mode = "fixed"
curvature = 0.1
dynamic_curvature = False
per_head_curvature = False
use_embedding_curvature = False
optimizer = "AdamW"
learning_rate = 5e-4
```

### Balanced quality/speed direction

Use the simpler mixed-curvature configuration plus the speed stack:

1. fused hyperbolic ops
2. `torch.compile(mode="default", fullgraph=False)`
3. Triton kernels for the core hyperbolic primitives

## Final Conclusion

The graph now supports four top-level conclusions:

1. Mixed-curvature does improve representation isotropy relative to a matched Euclidean baseline.
2. The original pretraining-speed gain is real, but regime-sensitive rather than universal.
3. That speed gain is not currently robust to Muon in the originally tested regime.
4. A simpler static-curvature configuration outperforms the earlier dynamic/per-head setup, and the path to reducing overhead is now explicit in the repo through the fused, compiled, cached, and Triton-based implementations.

The root should now be read as a completed abstract of the validation graph, including the later ablation and speed-optimization branches, not just the original March 20 validation pass.
