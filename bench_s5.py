"""
S5 benchmark: Triton kernels vs torch.jit.script fused ops.

Run:
    python bench_s5.py 2>&1 | tee /tmp/s5_bench.log
"""

import sys
import time
import torch

sys.path.insert(0, '/root/hyperbolic_transformer')

device = 'cuda'
B, T, C = 4, 128, 768

# ── correctness check helpers ─────────────────────────────────────────────

def check(name, a, b, tol=1e-4):
    err = (a - b).abs().max().item()
    status = "PASS" if err < tol else "FAIL"
    print(f"  [{status}] {name:20s}  max_err={err:.3e}")
    return err < tol


def bench(fn, *args, n_warmup=20, n_bench=200):
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_bench):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_bench * 1000  # ms


def main():
    print(f"PyTorch {torch.__version__}")
    try:
        import triton
        print(f"Triton  {triton.__version__}")
        triton_ok = True
    except ImportError:
        print("Triton  NOT available")
        triton_ok = False

    print(f"CUDA    {torch.version.cuda}")
    print(f"Device  {torch.cuda.get_device_name(0)}")
    print()

    torch.manual_seed(42)
    x = torch.randn(B, T, C, device=device, dtype=torch.float32) * 0.1
    v = torch.randn(B, T, C, device=device, dtype=torch.float32) * 0.01
    c_float = 0.1
    c_tensor = torch.tensor(c_float, device=device, dtype=torch.float32)

    from model_fused import mobius_addition, expmap, logmap
    from model_triton import mobius_add_triton, expmap_triton, logmap_triton

    x_flat = x.flatten(0, 1)
    v_flat = v.flatten(0, 1)

    # ── correctness ──────────────────────────────────────────────────────
    print("=== Correctness ===")
    ref_mob  = mobius_addition(x_flat, v_flat, c_tensor).view(B, T, C)
    ref_exp  = expmap(x_flat, v_flat, c_tensor).view(B, T, C)
    ref_log  = logmap(x_flat, v_flat, c_tensor).view(B, T, C)

    trit_mob = mobius_add_triton(x, v, c_float)
    trit_exp = expmap_triton(x, v, c_float)
    trit_log = logmap_triton(x, v, c_float)

    all_pass = True
    all_pass &= check("mobius_add", ref_mob, trit_mob)
    all_pass &= check("expmap",     ref_exp, trit_exp)
    all_pass &= check("logmap",     ref_log, trit_log)
    print()

    if not all_pass:
        print("WARNING: some correctness checks failed — speedups may be unreliable")

    # ── throughput ───────────────────────────────────────────────────────
    print("=== Throughput (ms per call, 200 iterations after 20 warmup) ===")
    print(f"  Shape: B={B} T={T} C={C}")
    print()

    t_mob_fused  = bench(mobius_addition, x_flat, v_flat, c_tensor)
    t_mob_triton = bench(mobius_add_triton, x, v, c_float)
    su_mob = t_mob_fused / t_mob_triton if t_mob_triton > 0 else float('nan')
    print(f"  mobius_add:  fused={t_mob_fused:.4f}ms  triton={t_mob_triton:.4f}ms  speedup={su_mob:.2f}x")

    t_exp_fused  = bench(expmap, x_flat, v_flat, c_tensor)
    t_exp_triton = bench(expmap_triton, x, v, c_float)
    su_exp = t_exp_fused / t_exp_triton if t_exp_triton > 0 else float('nan')
    print(f"  expmap:      fused={t_exp_fused:.4f}ms  triton={t_exp_triton:.4f}ms  speedup={su_exp:.2f}x")

    t_log_fused  = bench(logmap, x_flat, v_flat, c_tensor)
    t_log_triton = bench(logmap_triton, x, v, c_float)
    su_log = t_log_fused / t_log_triton if t_log_triton > 0 else float('nan')
    print(f"  logmap:      fused={t_log_fused:.4f}ms  triton={t_log_triton:.4f}ms  speedup={su_log:.2f}x")

    print()
    print("=== Summary ===")
    print(f"  triton_available = {triton_ok}")
    print(f"  correctness_pass = {all_pass}")
    print(f"  speedups: mobius_add={su_mob:.2f}x  expmap={su_exp:.2f}x  logmap={su_log:.2f}x")
    geomean = (su_mob * su_exp * su_log) ** (1.0 / 3.0)
    print(f"  geometric_mean_speedup = {geomean:.2f}x")
    print()
    print("Note: S3 (torch.compile) already fuses these ops via Inductor-generated Triton kernels.")
    print("Custom Triton kernels reduce Python launch overhead and allow hand-tuned blocking.")


if __name__ == '__main__':
    main()
