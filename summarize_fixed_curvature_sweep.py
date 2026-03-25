#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


STEP_PATTERN = re.compile(r"step (\d+): train loss ([0-9.]+), val loss ([0-9.]+)")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return {}


def json_safe(value: Any):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


def parse_history(log_path: Path) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    if not log_path.exists():
        return history
    with open(log_path) as handle:
        for line in handle:
            match = STEP_PATTERN.search(line)
            if not match:
                continue
            history.append(
                {
                    "step": int(match.group(1)),
                    "train_loss": float(match.group(2)),
                    "val_loss": float(match.group(3)),
                }
            )
    return history


def summarize_run(run_dir: Path) -> Optional[dict]:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return None

    metadata = read_json(metadata_path)
    summary_path = run_dir / "summary.json"
    existing_summary = read_json(summary_path)
    manifest = read_json(run_dir / "run_manifest.json")

    history_path = run_dir / "history.json"
    history = read_json(history_path)
    if not history:
        history = parse_history(run_dir / "train.log")
        with open(history_path, "w") as handle:
            json.dump(history, handle, indent=2)

    checkpoint_path = run_dir / "ckpt.pt"
    checkpoint = None
    checkpoint_error = None
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            checkpoint_error = str(exc)

    best_eval = min(history, key=lambda item: item["val_loss"]) if history else None
    final_eval = history[-1] if history else None

    summary = {
        "run_dir": str(run_dir),
        "history_path": str(history_path),
        "manifest_path": str(run_dir / "run_manifest.json") if manifest else None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path.exists() else None,
        "checkpoint_error": checkpoint_error,
        "success": checkpoint_path.exists(),
        "num_evals": len(history),
        "best_eval": best_eval,
        "final_eval": final_eval,
        "checkpoint_iter_num": checkpoint.get("iter_num") if checkpoint else None,
        "checkpoint_best_val_loss": checkpoint.get("best_val_loss") if checkpoint else None,
    }
    summary.update(existing_summary)
    summary.update(metadata)
    if manifest:
        summary["run_manifest"] = manifest

    if "sweep_parameter_sort" not in summary and "sweep_parameter_value" in summary:
        summary["sweep_parameter_sort"] = summary["sweep_parameter_value"]

    summary = json_safe(summary)
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def find_run_dirs(run_root: Path) -> List[Path]:
    return sorted(path.parent for path in run_root.rglob("run_metadata.json"))


def maybe_normalize_sort_key(value: Any):
    if isinstance(value, list):
        return tuple(maybe_normalize_sort_key(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, maybe_normalize_sort_key(item)) for key, item in value.items()))
    return value


def build_aggregate(summaries: List[dict]) -> dict:
    families = defaultdict(list)
    for summary in summaries:
        families[(summary.get("family"), summary.get("sweep_parameter_name"))].append(summary)

    aggregate_families = {}
    for (family_name, sweep_parameter_name), items in sorted(families.items()):
        sorted_items = sorted(
            items,
            key=lambda item: (
                maybe_normalize_sort_key(item.get("sweep_parameter_sort")),
                item.get("model_kind", ""),
            ),
        )
        aggregate_families[family_name] = {
            "sweep_parameter_name": sweep_parameter_name,
            "runs": sorted_items,
        }

    return {"runs": summaries, "families": aggregate_families}


def main():
    parser = argparse.ArgumentParser(description="Summarize fixed-curvature and scaling sweeps from logs and checkpoints.")
    parser.add_argument("--run-root", default="runs/fineweb_constant_curvature_refine")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    summaries = []
    for run_dir in find_run_dirs(run_root):
        summary = summarize_run(run_dir)
        if summary is not None:
            summaries.append(summary)

    aggregate = build_aggregate(summaries)
    aggregate = json_safe(aggregate)
    aggregate_path = run_root / "aggregate_summary.json"
    with open(aggregate_path, "w") as handle:
        json.dump(aggregate, handle, indent=2)

    print(json.dumps(aggregate, indent=2))
    print(f"Wrote aggregate summary to {aggregate_path}")


if __name__ == "__main__":
    main()
