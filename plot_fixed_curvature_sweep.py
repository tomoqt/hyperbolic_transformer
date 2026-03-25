#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLOR_MAP = {"baseline": "#1f77b4", "hyperbolic": "#d62728"}


def load_aggregate(run_root: Path) -> dict:
    aggregate_path = run_root / "aggregate_summary.json"
    if aggregate_path.exists():
        with open(aggregate_path) as handle:
            return json.load(handle)
    return {"runs": [], "families": {}}


def sort_key(value: Any):
    if isinstance(value, list):
        return tuple(sort_key(item) for item in value)
    return value


def label_for(value: Any) -> str:
    if isinstance(value, list):
        return "x".join(str(item) for item in value)
    return str(value)


def plot_best_vs_parameter(runs: List[dict], output_path: Path, sweep_parameter_name: str):
    grouped = defaultdict(list)
    for summary in runs:
        grouped[summary["model_kind"]].append(
            (
                summary.get("sweep_parameter_sort"),
                summary.get("best_eval", {}).get("val_loss"),
                summary.get("sweep_parameter_display", summary.get("sweep_parameter_value")),
            )
        )

    normalized_to_label = {}
    all_normalized = []
    for points in grouped.values():
        for sort_value, _, display in points:
            normalized = sort_key(sort_value)
            normalized_to_label[normalized] = display
            all_normalized.append(normalized)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    has_only_scalars = all(isinstance(value, (int, float)) for value in all_normalized)

    if has_only_scalars:
        for model_kind, points in sorted(grouped.items()):
            points.sort(key=lambda item: sort_key(item[0]))
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                color=COLOR_MAP.get(model_kind, None),
                label=model_kind,
            )
        if sweep_parameter_name == "curvature":
            ax.set_xscale("log")
    else:
        ordered_keys = sorted(normalized_to_label.keys())
        positions = {key: idx for idx, key in enumerate(ordered_keys)}
        for model_kind, points in sorted(grouped.items()):
            points.sort(key=lambda item: sort_key(item[0]))
            xs = [positions[sort_key(point[0])] for point in points]
            ys = [point[1] for point in points]
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                color=COLOR_MAP.get(model_kind, None),
                label=model_kind,
            )
        ax.set_xticks(range(len(ordered_keys)))
        ax.set_xticklabels([label_for(normalized_to_label[key]) for key in ordered_keys], rotation=30, ha="right")

    ax.set_xlabel(sweep_parameter_name.replace("_", " "))
    ax.set_ylabel("best val loss")
    ax.set_title(f"Best Validation Loss vs {sweep_parameter_name.replace('_', ' ').title()}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_curves(runs: List[dict], metric_key: str, output_path: Path, sweep_parameter_name: str):
    grouped = defaultdict(list)
    for summary in runs:
        grouped[summary.get("sweep_parameter_display", summary.get("sweep_parameter_value"))].append(summary)

    parameter_values = sorted(grouped.keys(), key=lambda value: sort_key(grouped[value][0].get("sweep_parameter_sort")))
    cols = 2
    rows = max(1, math.ceil(len(parameter_values) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)

    for ax in axes.flat:
        ax.set_visible(False)

    for idx, parameter_value in enumerate(parameter_values):
        ax = axes[idx // cols][idx % cols]
        ax.set_visible(True)
        runs_for_value = grouped[parameter_value]
        for summary in sorted(runs_for_value, key=lambda item: item["model_kind"]):
            history_path = Path(summary["history_path"])
            history = []
            if history_path.exists():
                with open(history_path) as handle:
                    history = json.load(handle)
            if not history:
                continue
            steps = [point["step"] for point in history]
            values = [point[metric_key] for point in history]
            ax.plot(
                steps,
                values,
                linewidth=2,
                color=COLOR_MAP.get(summary["model_kind"], None),
                label=summary["model_kind"],
            )
        ax.set_title(f"{sweep_parameter_name}={parameter_value}")
        ax.set_xlabel("step")
        ax.set_ylabel(metric_key.replace("_", " "))
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot fixed-curvature refinement and scaling sweeps.")
    parser.add_argument("--run-root", default="runs/fineweb_constant_curvature_refine")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    aggregate = load_aggregate(run_root)
    families = aggregate.get("families", {})

    if not families:
        raise SystemExit(f"No family summaries found under {run_root}. Run the summarizer first.")

    for family_name, payload in families.items():
        runs = payload.get("runs", [])
        sweep_parameter_name = payload.get("sweep_parameter_name", "parameter")
        if not runs:
            continue
        sanitized = sweep_parameter_name.replace(" ", "_")
        plot_best_vs_parameter(runs, run_root / f"best_val_vs_{sanitized}.png", sweep_parameter_name)
        histories_available = any(Path(summary["history_path"]).exists() for summary in runs)
        if histories_available:
            plot_curves(runs, "train_loss", run_root / f"train_loss_curves_by_{sanitized}.png", sweep_parameter_name)
            plot_curves(runs, "val_loss", run_root / f"val_loss_curves_by_{sanitized}.png", sweep_parameter_name)

    print(f"Wrote plots under {run_root}")


if __name__ == "__main__":
    main()
