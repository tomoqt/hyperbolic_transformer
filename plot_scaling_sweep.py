#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLOR_MAP = {"baseline": "#1f77b4", "mixed_curvature": "#d62728"}


def load_aggregate(run_root: Path):
    aggregate_path = run_root / "aggregate_summary.json"
    if not aggregate_path.exists():
        raise SystemExit(f"Missing aggregate summary at {aggregate_path}")
    return json.loads(aggregate_path.read_text())


def load_history_for_run(summary):
    history_path = summary.get("history_path")
    if history_path:
        path = Path(history_path)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass

    run_dir = summary.get("run_dir")
    if run_dir:
        path = Path(run_dir) / "history.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
    return []


def group_runs(runs):
    grouped = defaultdict(list)
    for summary in runs:
        grouped[summary.get("model_kind", "unknown")].append(summary)
    for model_kind in grouped:
        grouped[model_kind].sort(key=lambda item: item.get("scale_order", 0))
    return grouped


def sorted_scales(runs):
    return sorted(
        {
            (run.get("scale_order", 0), run.get("scale_label", str(run.get("scale_value"))))
            for run in runs
        },
        key=lambda item: item[0],
    )


def plot_line_metric(grouped, metric_key, ylabel, title, output_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_kind, summaries in grouped.items():
        x = [item.get("scale_order", 0) for item in summaries]
        y = [item.get(metric_key) for item in summaries]
        labels = [item.get("scale_label", str(item.get("scale_value"))) for item in summaries]
        if not any(value is not None for value in y):
            continue
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label=model_kind,
            color=COLOR_MAP.get(model_kind),
        )
        for xi, yi, label in zip(x, y, labels):
            if yi is not None:
                ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    ax.set_xlabel("scale")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks([item[0] for item in sorted_scales([run for runs in grouped.values() for run in runs])])
    ax.set_xticklabels([item[1] for item in sorted_scales([run for runs in grouped.values() for run in runs])])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_best_val(runs, output_path):
    grouped = group_runs(runs)
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_kind, summaries in grouped.items():
        x = [item.get("scale_order", 0) for item in summaries]
        y = [item.get("best_eval", {}).get("val_loss") for item in summaries]
        labels = [item.get("scale_label", str(item.get("scale_value"))) for item in summaries]
        if not any(value is not None for value in y):
            continue
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label=model_kind,
            color=COLOR_MAP.get(model_kind),
        )
        for xi, yi, label in zip(x, y, labels):
            if yi is not None:
                ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    ax.set_xlabel("scale")
    ax.set_ylabel("best val loss")
    ax.set_title("Best Validation Loss vs Scale")
    ax.set_xticks([item[0] for item in sorted_scales(runs)])
    ax.set_xticklabels([item[1] for item in sorted_scales(runs)])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_history_panels(runs, metric_key, ylabel, title, output_path):
    runs_by_scale = defaultdict(list)
    for run in runs:
        runs_by_scale[run.get("scale_label", str(run.get("scale_value")))].append(run)

    scale_items = sorted(runs_by_scale.items(), key=lambda item: item[1][0].get("scale_order", 0))
    if not scale_items:
        return

    cols = 2
    rows = (len(scale_items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for idx, (scale_label, scale_runs) in enumerate(scale_items):
        ax = axes[idx // cols][idx % cols]
        ax.set_visible(True)
        for run in sorted(scale_runs, key=lambda item: item.get("model_kind", "")):
            history = run.get("history") or load_history_for_run(run)
            if not history:
                continue
            steps = [point["step"] for point in history if metric_key in point]
            values = [point[metric_key] for point in history if metric_key in point]
            if not values:
                continue
            ax.plot(
                steps,
                values,
                label=run.get("model_kind", "run"),
                color=COLOR_MAP.get(run.get("model_kind")),
                linewidth=2,
            )
        ax.set_title(f"scale={scale_label}")
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_paired_deltas(aggregate, output_path):
    comparisons = aggregate.get("paired_comparisons", [])
    if not comparisons:
        return

    x = [item.get("scale_order", 0) for item in comparisons]
    labels = [item.get("scale_label", str(item.get("scale_order"))) for item in comparisons]
    y_val = [item.get("deltas", {}).get("best_val_loss") for item in comparisons]
    y_time = [item.get("deltas", {}).get("mean_train_step_ms_ratio") for item in comparisons]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.3)
    ax.plot(x, y_val, marker="o", linewidth=2, color="#d62728", label="val loss delta (mixed - baseline)")
    ax.set_xlabel("scale")
    ax.set_ylabel("val loss delta")
    ax.set_title("Mixed-curvature Delta vs Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(x, y_time, marker="s", linewidth=2, color="#1f77b4", label="train-step time ratio")
    ax2.set_ylabel("time ratio")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot context/model scaling sweep results.")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    aggregate = load_aggregate(run_root)
    runs = aggregate.get("runs", [])
    if not runs:
        raise SystemExit(f"No summarized runs found under {run_root}")

    grouped = group_runs(runs)

    plot_best_val(runs, run_root / "best_val_vs_scale.png")
    plot_line_metric(
        grouped,
        "mean_train_step_ms",
        "mean train step time (ms)",
        "Mean Train Step Time vs Scale",
        run_root / "mean_train_step_ms_vs_scale.png",
    )
    plot_line_metric(
        grouped,
        "mean_train_tokens_per_sec",
        "mean train tokens / sec",
        "Mean Training Throughput vs Scale",
        run_root / "mean_train_tokens_per_sec_vs_scale.png",
    )
    plot_history_panels(
        runs,
        "train_loss",
        "train loss",
        "Train Loss Curves by Scale",
        run_root / "train_loss_curves_by_scale.png",
    )
    plot_history_panels(
        runs,
        "val_loss",
        "val loss",
        "Validation Loss Curves by Scale",
        run_root / "val_loss_curves_by_scale.png",
    )
    plot_paired_deltas(aggregate, run_root / "paired_deltas_vs_scale.png")

    print(f"Wrote plots under {run_root}")


if __name__ == "__main__":
    main()
