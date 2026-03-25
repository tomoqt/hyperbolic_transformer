"""Benchmark S4 (curvature precompute) vs S3 baseline.

Tests curvature_mode='fixed' (maximum S4 benefit) and 'random' (learnable
parameters -- representative of typical training scenario).

Usage:
    python bench_s4.py 2>&1 | tee /tmp/s4_bench.log
"""
import time

import torch

device = 'cuda'


def bench(fn, n_warmup=10, n_bench=50):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_bench):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_bench * 1000  # ms


# ---------------------------------------------------------------------------
# S3 baseline
# ---------------------------------------------------------------------------
from model_compiled import GPT as GPT_S3, GPTConfig, get_compiled_model

print("=" * 60)
print("S4 Curvature Precompute Benchmark")
print("=" * 60)

for mode in ('fixed', 'random'):
    print(f"\n--- curvature_mode='{mode}' ---")

    cfg_kwargs = dict(
        block_size=256,
        vocab_size=50304,
        n_layer=6,
        n_head=6,
        n_embd=384,
        use_baseline_model=False,
        curvature_mode=mode,
        dynamic_curvature=False,
        per_head_curvature=True,
        bias=False,
        dropout=0.0,
    )
    # GPTConfig might not accept use_baseline_model; guard it.
    try:
        config = GPTConfig(**cfg_kwargs)
    except TypeError:
        cfg_kwargs.pop('use_baseline_model', None)
        config = GPTConfig(**cfg_kwargs)

    if mode == 'fixed':
        config.curvature = 0.5

    x = torch.randint(0, 50304, (2, 256), device=device)
    y = torch.randint(0, 50304, (2, 256), device=device)

    # --- S3: torch.compile on vanilla model.GPT ---
    model_s3 = GPT_S3(config).to(device)
    model_s3.eval()
    model_s3_c = get_compiled_model(model_s3)
    t_s3 = bench(lambda: model_s3_c(x, y))
    print(f"  S3 baseline (compile only):             {t_s3:.2f} ms")

    # --- S4: model_precompute.GPT + update_curvature_cache + compile ---
    from model_precompute import GPT as GPT_S4
    model_s4 = GPT_S4(config).to(device)
    model_s4.eval()
    model_s4.update_curvature_cache()
    model_s4_c = torch.compile(model_s4, mode='default', fullgraph=False)
    t_s4_compiled = bench(lambda: model_s4_c(x, y))
    print(f"  S4 (precompute + compile):              {t_s4_compiled:.2f} ms  "
          f"({t_s3 / t_s4_compiled:.2f}x vs S3)")

    # --- S4 eager (no compile) to show raw precompute benefit ---
    model_s4_eager = GPT_S4(config).to(device)
    model_s4_eager.eval()
    model_s4_eager.update_curvature_cache()
    t_s4_eager = bench(lambda: model_s4_eager(x, y))
    print(f"  S4 eager (precompute, no compile):      {t_s4_eager:.2f} ms")

    # --- S3-equivalent eager (no compile) for fair eager comparison ---
    model_s3_eager = GPT_S3(config).to(device)
    model_s3_eager.eval()
    t_s3_eager = bench(lambda: model_s3_eager(x, y))
    print(f"  S3 eager (no compile, no precompute):   {t_s3_eager:.2f} ms  "
          f"(S4-eager speedup: {t_s3_eager / t_s4_eager:.2f}x)")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
