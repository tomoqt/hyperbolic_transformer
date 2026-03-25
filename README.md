# Mixed-Curvature Transformers

Mixed-curvature transformer experiments built on [nanoGPT](https://github.com/karpathy/nanoGPT). The project asks whether adding hyperbolic structure inside the residual stream changes representation geometry in a useful way, and whether that geometric change can translate into faster optimization.

## Motivation

Standard transformer training assumes a Euclidean representation space everywhere. The working hypothesis here is that hidden states are anisotropic enough that a mixed-curvature parameterization can be a better fit: use Euclidean machinery where it is convenient, but let the residual stream operate in a curved space when that improves geometry or optimization.

This repo grew out of two concrete questions:

1. Does the mixed-curvature model produce more isotropic hidden representations than a matched Euclidean baseline?
2. Does it train faster in practice, and if so, is that effect broad, schedule-sensitive, or optimizer-dependent?

## What Model Is Actually Being Tested?

The validated model is **mixed-curvature**, not fully hyperbolic end to end.

- Token and positional embeddings are combined in Euclidean space.
- In the main validation configs, `use_embedding_curvature=False`.
- Transformer blocks apply hyperbolic residual-style updates internally.
- The final layer norm and LM head remain Euclidean.

The initial March 2026 validation used:

- `curvature_mode='random'`
- `dynamic_curvature=True`
- `per_head_curvature=True`
- `use_embedding_curvature=False`

Later ablations found that a simpler configuration works better.

## Main Results (March 2026)

### Initial validation

All main validation runs used `1500` optimizer steps per run.

- Shakespeare budget: `3072` tokens/step, `4.608M` tokens/run
- FineWeb budget: `4096` tokens/step, `6.144M` tokens/run

Under the original AdamW setup:

- **Isotropy**: mixed-curvature improves isotropy relative to baseline.
  - Shakespeare: the gain appears mainly in deeper layers.
  - FineWeb: the gain appears across all six probed layers.
- **Speed**: the optimization gain is real but regime-sensitive.
  - Shakespeare LR sweep: mixed-curvature beats baseline across the full sweep.
  - FineWeb LR sweep: mixed-curvature wins at `5e-4` and `1e-3`, while baseline wins at `1e-4` and `2e-4`.

Under the Muon extension:

- **Isotropy survives**: deeper-layer isotropy gains remain visible on both Shakespeare and FineWeb.
- **Speed does not survive yet**: the mixed-curvature Muon speed branch diverged at the Shakespeare gate under the original coarse sweep settings, so the downstream FineWeb speed branch was intentionally canceled.

### Systematic ablations (A0-A7)

The ablation suite isolates which parts of the mixed-curvature design actually matter.

Most important takeaways:

- Best overall run: **fixed curvature `c=0.1`**
- Dynamic curvature: neutral in this setup
- Per-head curvature: tiny gain, not worth the extra complexity/overhead
- Embedding curvature: slightly worse than leaving embeddings Euclidean
- Muon: unstable in the tested regime

Best quality configuration from the graph:

```python
curvature_mode = "fixed"
curvature = 0.1
dynamic_curvature = False
per_head_curvature = False
use_embedding_curvature = False
optimizer = "AdamW"
learning_rate = 5e-4
```

This configuration reached `val_loss = 5.5822` on FineWeb versus the Euclidean baseline at `6.022`.

### Speed optimization branch

The initial matched FineWeb `5e-4` rerun showed a substantial overhead for mixed-curvature:

- end-to-end slowdown: about `3.18x`
- train-step-only slowdown: about `5.86x`

The March 25 speed branch then attacked that overhead directly:

- `model_fused.py`: TorchScript-fused hyperbolic ops
- `model_compiled.py`: `torch.compile(mode="default", fullgraph=False)` wrappers
- `model_precompute.py`: cached static curvature transforms
- `model_triton.py`: custom Triton kernels for `mobius_addition`, `expmap`, and `logmap`
- `bench_s4.py`, `bench_s5.py`: reproducible benchmark harnesses

Key benchmark findings:

- TorchScript fusion cut kernel launches by about `71%` in the profiling branch
- `torch.compile(..., mode="default")` gave the biggest model-level gain in the benchmark harness
- Triton kernels improved primitive throughput:
  - `mobius_addition`: `1.70x`
  - `expmap`: `2.14x`
  - `logmap`: `3.42x`

These speed modules are included in the public repo as benchmarked implementation paths. The default training path in `train.py` still uses the standard model unless you explicitly swap in one of the optimized variants.

## Repository Layout

- `train.py`: main training entrypoint, logging, checkpointing, Muon support
- `model.py`: reference mixed-curvature model
- `model_baseline.py`: Euclidean baseline model
- `model_fused.py`: TorchScript-fused hyperbolic ops
- `model_compiled.py`: compile helpers and compiled op wrappers
- `model_precompute.py`: cached-curvature model variant
- `model_triton.py`: Triton kernels for the core hyperbolic ops
- `run_*.sh`: experiment launchers used in the validation graph
- `runs/`: run logs, summaries, histories, plots
- `analysis_out/`: isotropy analyses and root-level markdown summaries

The current graph-level abstract is mirrored in:

- `analysis_out/root_validation_readme_20260320.md`

## Data Preparation

Shakespeare-char:

```sh
python data/shakespeare_char/prepare.py
```

FineWeb:

```sh
python data/fineweb/prepare.py
```

## Reproducing Key Experiments

Install the basic dependencies:

```sh
pip install torch numpy matplotlib
pip install datasets tiktoken
```

Representative runs:

```sh
# AdamW LR sweeps
./run_shakespeare_lr_sweep.sh
./run_fineweb_lr_sweep.sh

# Muon LR sweep
./run_shakespeare_lr_sweep_muon.sh

# Isotropy studies
./run_shakespeare_isotropy.sh
./run_fineweb_isotropy.sh
./run_shakespeare_isotropy_muon.sh
./run_fineweb_isotropy_muon.sh

# Speed benchmarks
python bench_s4.py
python bench_s5.py
```

Plots and summaries can be regenerated from saved run directories:

```sh
python summarize_lr_sweep.py --run_root runs/shakespeare_lr_sweep_modal_20260320
python plot_lr_sweep.py --run_root runs/shakespeare_lr_sweep_modal_20260320
python analyze_representations.py
```

## Current Recommendation

If the goal is best validation quality in the current setup, start with the fixed-curvature configuration above.

If the goal is to keep the original mixed-curvature idea while reducing overhead, use the speed branch as the implementation roadmap:

1. fused ops
2. `torch.compile(mode="default", fullgraph=False)`
3. Triton kernels for the core hyperbolic primitives

## Open Problems

- Make the mixed-curvature speed story robust under Muon
- Determine whether the instability is optimizer-driven, geometry-driven, or specifically tied to Mobius residual updates
- Validate the speed stack end-to-end in the main training loop, not only in benchmark harnesses
- Push beyond the current negative-curvature-only stereographic setup

## Acknowledgements

- Original code based on [nanoGPT](https://github.com/karpathy/nanoGPT)
- Geometric optimization inspiration from [Muon](https://github.com/KellerJordan/Muon)
