#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_histories(run_root: Path):
    grouped = {}
    for child in sorted(run_root.iterdir()):
        if not child.is_dir():
            continue
        summary_path = child / "summary.json"
        history_path = child / "history.json"
        if not summary_path.exists() or not history_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        history = json.loads(history_path.read_text())
        grouped.setdefault(summary["learning_rate"], {})[summary["model_kind"]] = {
            "summary": summary,
            "history": history,
        }
    return grouped


def first_crossing(history, threshold):
    for point in history:
        if point["val_loss"] <= threshold:
            return point
    return None


def plot_crossings(entries, output_path: Path):
    xs = [float(entry["learning_rate"]) for entry in entries]
    baseline_steps = [entry["baseline"]["shared_target_step"] for entry in entries]
    hyperbolic_steps = [entry["hyperbolic"]["shared_target_step"] for entry in entries]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, baseline_steps, marker="o", linewidth=2, label="baseline", color="#1f77b4")
    ax.plot(xs, hyperbolic_steps, marker="o", linewidth=2, label="hyperbolic", color="#d62728")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("step to shared target loss")
    ax.set_title("Shared-Target Crossing Steps vs Learning Rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze matched target-loss crossing in an LR sweep.")
    parser.add_argument("--run-root", default="runs/shakespeare_lr_sweep")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    grouped = load_histories(run_root)
    entries = []
    for learning_rate in sorted(grouped.keys(), key=float):
        pair = grouped[learning_rate]
        if "baseline" not in pair or "hyperbolic" not in pair:
            continue
        baseline = pair["baseline"]
        hyperbolic = pair["hyperbolic"]
        shared_target = max(
            baseline["summary"]["best_eval"]["val_loss"],
            hyperbolic["summary"]["best_eval"]["val_loss"],
        )
        baseline_crossing = first_crossing(baseline["history"], shared_target)
        hyperbolic_crossing = first_crossing(hyperbolic["history"], shared_target)
        if baseline_crossing is None or hyperbolic_crossing is None:
            continue
        entries.append(
            {
                "learning_rate": learning_rate,
                "shared_target_val_loss": shared_target,
                "baseline": {
                    "best_val_loss": baseline["summary"]["best_eval"]["val_loss"],
                    "shared_target_step": baseline_crossing["step"],
                },
                "hyperbolic": {
                    "best_val_loss": hyperbolic["summary"]["best_eval"]["val_loss"],
                    "shared_target_step": hyperbolic_crossing["step"],
                },
                "step_advantage_h_minus_b": hyperbolic_crossing["step"] - baseline_crossing["step"],
                "faster_model": (
                    "hyperbolic"
                    if hyperbolic_crossing["step"] < baseline_crossing["step"]
                    else "baseline"
                    if hyperbolic_crossing["step"] > baseline_crossing["step"]
                    else "tie"
                ),
            }
        )

    output_path = Path(args.output) if args.output else run_root / "target_crossing_summary.json"
    payload = {"entries": entries}
    output_path.write_text(json.dumps(payload, indent=2))
    plot_crossings(entries, run_root / "shared_target_crossing_steps.png")
    print(json.dumps(payload, indent=2))
    print(f"Wrote target-crossing summary to {output_path}")


if __name__ == "__main__":
    main()
