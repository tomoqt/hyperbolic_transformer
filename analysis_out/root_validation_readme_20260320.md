# Mixed-Curvature Transformer Validation

## Motivation

This project tests whether adding hyperbolic structure inside a small GPT-style transformer changes representation geometry in a useful way and whether that geometric change translates into faster optimization. The motivating idea is that a mixed-curvature parameterization may produce more isotropic hidden representations and, in the right regime, a more favorable optimization landscape than a matched Euclidean baseline.

## Questions

1. Does the mixed-curvature model increase representation isotropy relative to a matched baseline?
2. Does it accelerate pretraining in this setup, and if so, is that effect broad, schedule-sensitive, or optimizer-dependent?

## Architecture Clarification

The current experimental model is not fully hyperbolic end to end. It is a mixed-curvature model with the following structure:

- Token and positional embeddings are added in Euclidean space.
- `use_embedding_curvature` is disabled in the active validation configs.
- Transformer blocks apply hyperbolic residual-style updates internally.
- The final layer norm and LM head remain Euclidean.

The active mixed-curvature branch uses dynamic, per-head curvature with random initialization:

- `curvature_mode = random`
- `dynamic_curvature = True`
- `per_head_curvature = True`
- `use_embedding_curvature = False`

So the accurate description of the current experimental model is mixed-curvature rather than fully hyperbolic.

## Training Budget and Measured Overhead

All validation sweeps and isotropy runs use `1500` optimizer steps per run.

- Shakespeare tokens per step: `3072`
- Shakespeare tokens per run: `4,608,000`
- Fineweb tokens per step: `4096`
- Fineweb tokens per run: `6,144,000`

Measured end-to-end overhead on the matched Fineweb `5e-4` rerun, using the same host:

- Baseline mean logged step time: `461.88 ms`
- Mixed-curvature mean logged step time: `1466.89 ms`
- End-to-end slowdown: about `3.18x`
- Baseline non-eval mean step time: `121.78 ms`
- Mixed-curvature non-eval mean step time: `713.21 ms`
- Train-step-only slowdown: about `5.86x`

## Main Findings

### 1. Representation isotropy is a real effect

Under the original AdamW-style optimizer setup:

- Shakespeare isotropy improved mainly in deeper layers.
- Fineweb isotropy improved across all six probed layers.

So the isotropy hypothesis is supported in the original setup, with the effect becoming cleaner on the larger-data Fineweb regime.

### 2. Pretraining speed improved in the original setup, but only in a specific regime

Under the original optimizer:

- Shakespeare coarse LR sweep: mixed-curvature beat baseline across the full swept range.
- Fineweb coarse LR sweep: baseline won at `1e-4` and `2e-4`, while mixed-curvature won at `5e-4` and `1e-3`.

Interpretation: the optimization-speed signal is real in this setup, but it is schedule-sensitive rather than universal. The advantage appears concentrated in a higher-learning-rate band on Fineweb.

## Muon Extension

The Muon extension was added to answer a narrower question: does the mixed-curvature effect survive a different optimizer, or was the earlier result tightly coupled to the original optimization setup?

### Muon isotropy

The isotropy signal survives Muon, but with a different layer profile.

Shakespeare Muon isotropy:

- Stable setting: `learning_rate=2e-4`, `muon_lr_ratio=1`
- Early layers were worse or near-neutral.
- Deeper layers improved, strongest at layer `5` with `normalized_spectral_entropy_delta=+0.1674` and `effective_rank_delta=+33.50`.

Fineweb Muon isotropy:

- Stable setting: `learning_rate=2e-4`, `muon_lr=2e-4`, `muon_lr_ratio=1`
- Layers `0` and `1` regressed.
- Layers `2` through `5` improved progressively.
- Strongest result was layer `5`: `normalized_spectral_entropy_delta=+0.1787`, `effective_rank_delta=+53.34`, `participation_ratio_delta=+5.26`, `top1_share_delta=-0.0155`.

Interpretation: Muon does not erase the geometry effect. The mixed-curvature model still becomes more isotropic in deeper layers, but the Muon version is less uniformly positive than the original Fineweb result.

### Muon speed

The Muon speed branch did not reproduce the original acceleration story.

Shakespeare Muon coarse LR sweep:

- Baseline remained stable at every swept learning rate.
- Mixed-curvature diverged to `NaN` at every swept learning rate under the original Muon sweep setting `muon_lr_ratio=3`.

That result closes the Muon speed branch as a stability failure rather than an acceleration win. Because the small-scale Shakespeare gate already falsified that regime, the downstream Fineweb Muon speed subtree was intentionally skipped instead of spending compute on a larger version of a failed setting.

Interpretation: the mixed-curvature speed effect is not yet optimizer-robust. Under Muon, a stability-recovery branch would be the right next step, not a blind Fineweb sweep at the same failing settings.

## Current Conclusion

The graph now supports three high-level conclusions:

1. The mixed-curvature architecture does improve representation isotropy, and that conclusion survives both datasets and both optimizer families, although the exact layer profile changes.
2. The original pretraining-speed gain is real but regime-dependent: it is broad on Shakespeare and concentrated in a higher-LR band on Fineweb.
3. That speed gain is not currently robust to Muon. Under the original Muon sweep settings, the mixed-curvature model becomes unstable before a larger-data Muon speed comparison is justified.

## Plot and Artifact Coverage

Plots are attached both at the experiment-node level and at the root:

- coarse LR sweep `best_val_vs_lr` plots,
- layer-wise isotropy metric panels,
- layer-wise isotropy delta panels,
- and non-checkpoint bundles containing logs, manifests, histories, and analysis reports for the newer Muon nodes.

One limitation remains for some of the earliest coarse sweeps: not every legacy run preserved full per-step histories, so some older nodes support aggregate and best-vs-LR plots but not reconstructed full loss curves. The later follow-up and Muon nodes do retain those richer artifacts where noted.

## Graph State

The Muon comparison extension is now logically closed:

- Muon isotropy branch: completed on Shakespeare and Fineweb.
- Muon speed branch: completed at the Shakespeare gate, with the Fineweb descendants committed as intentionally skipped because the upstream regime was already falsified.

The root node should now be read as an abstract of completed results rather than a live run-status dashboard.
