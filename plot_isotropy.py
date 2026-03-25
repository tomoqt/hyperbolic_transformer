#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = [
    ("normalized_spectral_entropy", "Normalized Spectral Entropy"),
    ("effective_rank", "Effective Rank"),
    ("participation_ratio", "Participation Ratio"),
    ("top1_share", "Top-1 Share"),
    ("condition_number", "Condition Number"),
]


def load_report(path: Path):
    return json.loads(path.read_text())


def get_layers(report: dict):
    return sorted((int(layer) for layer in report["layers"].keys()))


def get_metric_values(report: dict, metric_key: str, layers):
    return [report["layers"][str(layer)][metric_key] for layer in layers]


def plot_metric_panels(baseline: dict, hyperbolic: dict, output_path: Path):
    layers = get_layers(baseline)
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), squeeze=False)
    color_map = {"baseline": "#1f77b4", "hyperbolic": "#d62728"}

    for ax in axes.flat:
        ax.set_visible(False)

    for idx, (metric_key, label) in enumerate(METRICS):
        ax = axes[idx // 2][idx % 2]
        ax.set_visible(True)
        ax.plot(
            layers,
            get_metric_values(baseline, metric_key, layers),
            marker="o",
            linewidth=2,
            label="baseline",
            color=color_map["baseline"],
        )
        ax.plot(
            layers,
            get_metric_values(hyperbolic, metric_key, layers),
            marker="o",
            linewidth=2,
            label="hyperbolic",
            color=color_map["hyperbolic"],
        )
        if metric_key == "condition_number":
            ax.set_yscale("log")
        ax.set_title(label)
        ax.set_xlabel("layer")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_delta_panels(summary: dict, output_path: Path):
    layers = [int(layer) for layer in summary["common_layers"]]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), squeeze=False)

    for ax in axes.flat:
        ax.set_visible(False)

    for idx, (metric_key, label) in enumerate(METRICS):
        ax = axes[idx // 2][idx % 2]
        ax.set_visible(True)
        deltas = [summary["layer_deltas"][str(layer)][f"{metric_key}_delta"] for layer in layers]
        colors = ["#d62728" if value >= 0 else "#1f77b4" for value in deltas]
        ax.bar(layers, deltas, color=colors, alpha=0.85)
        ax.axhline(0.0, color="#333333", linewidth=1)
        if metric_key == "condition_number":
            ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(f"{label} Delta")
        ax.set_xlabel("layer")
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot isotropy metric comparisons by layer.")
    parser.add_argument("--analysis-dir", required=True)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    summary = load_report(analysis_dir / "summary.json")
    baseline = load_report(analysis_dir / "baseline_isotropy.json")
    hyperbolic = load_report(analysis_dir / "hyperbolic_isotropy.json")

    plot_metric_panels(baseline, hyperbolic, analysis_dir / "isotropy_metrics_by_layer.png")
    plot_delta_panels(summary, analysis_dir / "isotropy_deltas_by_layer.png")

    print(f"Wrote isotropy plots under {analysis_dir}")


if __name__ == "__main__":
    main()
