#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot summary diagnostics for instability investigations.")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    aggregate_path = run_root / "aggregate_summary.json"
    if not aggregate_path.exists():
        raise SystemExit(f"Missing aggregate summary at {aggregate_path}")

    aggregate = json.loads(aggregate_path.read_text())
    runs = aggregate.get("runs", [])
    if not runs:
        raise SystemExit(f"No runs summarized under {run_root}")

    case_names = [run["case_name"] for run in runs]
    first_nonfinite = [
        run["first_nonfinite_iter"] if run.get("first_nonfinite_iter") is not None else run.get("run_manifest", {}).get("run", {}).get("max_iters", 0)
        for run in runs
    ]
    best_vals = [
        run.get("best_eval", {}).get("val_loss")
        if run.get("best_eval") is not None
        else None
        for run in runs
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d62728" if run.get("first_nonfinite_iter") is not None else "#2ca02c" for run in runs]
    ax.bar(case_names, first_nonfinite, color=colors)
    ax.set_ylabel("first nonfinite iter (or max_iters if stable)")
    ax.set_title("Instability Onset by Diagnostic Case")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(run_root / "first_nonfinite_iter_by_case.png", dpi=160)
    plt.close(fig)

    finite_best = [(name, val) for name, val in zip(case_names, best_vals) if val is not None]
    if finite_best:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar([name for name, _ in finite_best], [val for _, val in finite_best], color="#1f77b4")
        ax.set_ylabel("best val loss")
        ax.set_title("Best Validation Loss by Diagnostic Case")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(run_root / "best_val_loss_by_case.png", dpi=160)
        plt.close(fig)

    print(f"Wrote plots under {run_root}")


if __name__ == "__main__":
    main()
