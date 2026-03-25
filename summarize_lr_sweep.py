#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

import torch


STEP_PATTERN = re.compile(r"step (\d+): train loss ([0-9.]+), val loss ([0-9.]+)")
RUN_DIR_PATTERN = re.compile(r"(?P<model_kind>baseline|hyperbolic)_lr_(?P<lr_tag>.+)")


def to_jsonable(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def restore_learning_rate(lr_tag: str) -> str:
    sci_match = re.fullmatch(r"([0-9]+(?:_[0-9]+)?)e_([0-9]+)", lr_tag)
    if sci_match:
        base = sci_match.group(1).replace("_", ".")
        exponent = sci_match.group(2)
        return f"{base}e-{exponent}"
    return lr_tag.replace("__", ".")


def summarize_run(run_dir: Path) -> Optional[dict]:
    match = RUN_DIR_PATTERN.fullmatch(run_dir.name)
    if not match:
        return None

    log_path = run_dir / "train.log"
    checkpoint_path = run_dir / "ckpt.pt"
    history = []
    if log_path.exists():
        with open(log_path) as handle:
            for line in handle:
                step_match = STEP_PATTERN.search(line)
                if step_match:
                    history.append(
                        {
                            "step": int(step_match.group(1)),
                            "train_loss": float(step_match.group(2)),
                            "val_loss": float(step_match.group(3)),
                        }
                    )

    checkpoint = None
    checkpoint_error = None
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            checkpoint_error = str(exc)

    metadata = {}
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as handle:
            metadata = json.load(handle)

    existing_summary = {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as handle:
            existing_summary = json.load(handle)

    manifest = {}
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as handle:
            manifest = json.load(handle)

    history_path = run_dir / "history.json"
    with open(history_path, "w") as handle:
        json.dump(history, handle, indent=2)

    summary = {
        "model_kind": match.group("model_kind"),
        "learning_rate": restore_learning_rate(match.group("lr_tag")),
        "run_dir": str(run_dir),
        "runtime_seconds": None,
        "success": checkpoint_path.exists(),
        "num_evals": len(history),
        "history_path": str(history_path),
        "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "best_eval": min(history, key=lambda item: item["val_loss"]) if history else None,
        "final_eval": history[-1] if history else None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path.exists() else None,
        "checkpoint_iter_num": to_jsonable(checkpoint.get("iter_num")) if checkpoint else None,
        "checkpoint_best_val_loss": to_jsonable(checkpoint.get("best_val_loss")) if checkpoint else None,
        "checkpoint_error": checkpoint_error,
    }
    summary.update(existing_summary)
    summary.update(metadata)
    if manifest:
        summary["run_manifest"] = manifest

    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Summarize Shakespeare LR sweep runs from logs and checkpoints.")
    parser.add_argument("--run-root", default="runs/shakespeare_lr_sweep")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    summaries = []
    for child in sorted(run_root.iterdir()):
        if not child.is_dir():
            continue
        summary = summarize_run(child)
        if summary is not None:
            summaries.append(summary)

    aggregate = {"runs": summaries}
    aggregate_path = run_root / "aggregate_summary.json"
    with open(aggregate_path, "w") as handle:
        json.dump(aggregate, handle, indent=2)

    print(json.dumps(aggregate, indent=2))
    print(f"Wrote aggregate summary to {aggregate_path}")


if __name__ == "__main__":
    main()
