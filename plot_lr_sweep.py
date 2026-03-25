#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STEP_RE = re.compile(r"step\s+(\d+):\s+train loss\s+([0-9.]+),\s+val loss\s+([0-9.]+)")


def infer_summary_from_dir(run_dir: Path):
    parts = run_dir.name.split("_lr_")
    if len(parts) != 2:
        raise ValueError(f"Could not infer run metadata from {run_dir}")
    model_kind, lr_tag = parts
    learning_rate = lr_tag.replace("_", "-").replace("--", "_")
    # Handle tags such as 5e_4 -> 5e-4 and 1e_3 -> 1e-3.
    learning_rate = learning_rate.replace("e-", "e").replace("e_", "e-")
    return {
        "model_kind": model_kind,
        "learning_rate": learning_rate,
        "run_dir": str(run_dir),
    }


def parse_history_from_log(log_path: Path):
    history = []
    if not log_path.exists():
        return history
    for line in log_path.read_text().splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        step, train_loss, val_loss = match.groups()
        history.append(
            {
                "step": int(step),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
        )
    return history


def load_history_for_summary(run_root: Path, summary: dict):
    run_dir_name = Path(summary.get("run_dir", "")).name
    if not run_dir_name:
        return []

    local_run_dir = run_root / run_dir_name
    history_path = local_run_dir / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        if history:
            return history

    return parse_history_from_log(local_run_dir / "train.log")


def load_runs(run_root: Path):
    runs = []
    for child in sorted(run_root.iterdir()):
        if not child.is_dir():
            continue
        summary_path = child / "summary.json"
        history_path = child / "history.json"
        log_path = child / "train.log"
        if not summary_path.exists() and not log_path.exists():
            continue
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
        else:
            summary = infer_summary_from_dir(child)
        history = []
        if history_path.exists():
            history = json.loads(history_path.read_text())
        if not history:
            history = parse_history_from_log(log_path)
        if not history:
            continue
        if not summary.get("best_eval"):
            best_point = min(history, key=lambda item: item["val_loss"])
            summary = {
                **summary,
                "best_eval": {
                    "step": best_point["step"],
                    "val_loss": best_point["val_loss"],
                    "train_loss": best_point["train_loss"],
                },
            }
        runs.append((summary, history))
    if runs:
        return runs

    aggregate_path = run_root / "aggregate_summary.json"
    if not aggregate_path.exists():
        return runs

    aggregate = json.loads(aggregate_path.read_text())
    for summary in aggregate.get("runs", []):
        history = load_history_for_summary(run_root, summary)
        if history and not summary.get("best_eval"):
            best_point = min(history, key=lambda item: item["val_loss"])
            summary = {
                **summary,
                "best_eval": {
                    "step": best_point["step"],
                    "val_loss": best_point["val_loss"],
                    "train_loss": best_point["train_loss"],
                },
            }
        runs.append((summary, history))
    return runs


def sort_key(summary):
    return (float(summary["learning_rate"]), summary["model_kind"])


def plot_curves(runs, metric_key: str, output_path: Path):
    runs = sorted(runs, key=lambda item: sort_key(item[0]))
    learning_rates = sorted({summary["learning_rate"] for summary, _ in runs}, key=float)
    cols = 2
    rows = math.ceil(len(learning_rates) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
    color_map = {"baseline": "#1f77b4", "hyperbolic": "#d62728"}

    for ax in axes.flat:
        ax.set_visible(False)

    for idx, lr in enumerate(learning_rates):
        ax = axes[idx // cols][idx % cols]
        ax.set_visible(True)
        for summary, history in runs:
            if summary["learning_rate"] != lr:
                continue
            steps = [point["step"] for point in history]
            values = [point[metric_key] for point in history]
            ax.plot(
                steps,
                values,
                label=summary["model_kind"],
                color=color_map.get(summary["model_kind"], None),
                linewidth=2,
            )
        ax.set_title(f"lr={lr}")
        ax.set_xlabel("step")
        ax.set_ylabel(metric_key.replace("_", " "))
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_best_vs_lr(runs, output_path: Path):
    runs = sorted(runs, key=lambda item: sort_key(item[0]))
    grouped = {}
    for summary, _ in runs:
        grouped.setdefault(summary["model_kind"], []).append(
            (float(summary["learning_rate"]), summary["best_eval"]["val_loss"])
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    color_map = {"baseline": "#1f77b4", "hyperbolic": "#d62728"}
    for model_kind, pairs in grouped.items():
        pairs.sort()
        ax.plot(
            [item[0] for item in pairs],
            [item[1] for item in pairs],
            marker="o",
            linewidth=2,
            label=model_kind,
            color=color_map.get(model_kind, None),
        )

    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("best val loss")
    ax.set_title("Best Validation Loss vs Learning Rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot train/val loss curves for an LR sweep.")
    parser.add_argument("--run-root", default="runs/shakespeare_lr_sweep")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    runs = load_runs(run_root)
    if not runs:
        raise SystemExit(f"No run summaries with histories found under {run_root}")

    runs_with_history = [item for item in runs if item[1]]
    if runs_with_history:
        plot_curves(runs_with_history, "train_loss", run_root / "train_loss_curves_by_lr.png")
        plot_curves(runs_with_history, "val_loss", run_root / "val_loss_curves_by_lr.png")
    plot_best_vs_lr(runs, run_root / "best_val_vs_lr.png")

    print(f"Wrote plots under {run_root}")


if __name__ == "__main__":
    main()
