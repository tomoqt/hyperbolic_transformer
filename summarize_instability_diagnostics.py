#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return {}
    with open(path) as handle:
        return json.load(handle)


def load_jsonl(path: Path):
    if not path.exists():
        return []
    records = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def is_finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def summarize_case(run_dir: Path):
    metadata = load_json(run_dir / "run_metadata.json")
    manifest = load_json(run_dir / "run_manifest.json")
    eval_history = load_json(run_dir / "eval_history.json")
    metrics = load_jsonl(run_dir / "metrics.jsonl")

    finite_evals = [
        record
        for record in eval_history
        if is_finite_number(record.get("train_loss")) and is_finite_number(record.get("val_loss"))
    ]
    best_eval = min(finite_evals, key=lambda record: record["val_loss"]) if finite_evals else None
    final_eval = finite_evals[-1] if finite_evals else None

    first_nonfinite_iter = None
    first_nonfinite_source = None
    first_nonfinite_parameter = None
    last_finite_iter = None

    for record in metrics:
        if record.get("event") == "train_step":
            loss = record.get("loss")
            if is_finite_number(loss):
                last_finite_iter = record.get("iter")
            elif first_nonfinite_iter is None:
                first_nonfinite_iter = record.get("iter")
                first_nonfinite_source = "train_loss"

            optimizer_debug = record.get("optimizer_debug") or {}
            for source_key in ("after_adamw", "after_muon"):
                source_payload = optimizer_debug.get(source_key) or {}
                if source_payload.get("nonfinite_parameter_count", 0) > 0 and first_nonfinite_iter is None:
                    first_nonfinite_iter = record.get("iter")
                    first_nonfinite_source = source_key
                    first_nonfinite_parameter = source_payload.get("first_nonfinite_parameter")
                    break
        elif record.get("event") == "eval":
            train_loss = record.get("train_loss")
            val_loss = record.get("val_loss")
            if is_finite_number(train_loss) and is_finite_number(val_loss):
                last_finite_iter = record.get("iter")
            elif first_nonfinite_iter is None:
                first_nonfinite_iter = record.get("iter")
                first_nonfinite_source = "eval"

    summary = {
        "case_name": metadata.get("case_name", run_dir.name),
        "run_dir": str(run_dir),
        "success": first_nonfinite_iter is None,
        "first_nonfinite_iter": first_nonfinite_iter,
        "first_nonfinite_source": first_nonfinite_source,
        "first_nonfinite_parameter": first_nonfinite_parameter,
        "last_finite_iter": last_finite_iter,
        "num_eval_records": len(eval_history),
        "best_eval": best_eval,
        "final_eval": final_eval,
        "manifest_path": str(run_dir / "run_manifest.json") if (run_dir / "run_manifest.json").exists() else None,
        "metrics_path": str(run_dir / "metrics.jsonl") if (run_dir / "metrics.jsonl").exists() else None,
        "eval_history_path": str(run_dir / "eval_history.json") if (run_dir / "eval_history.json").exists() else None,
        "checkpoint_path": str(run_dir / "ckpt.pt") if (run_dir / "ckpt.pt").exists() else None,
    }
    summary.update(metadata)
    if manifest:
        summary["run_manifest"] = manifest

    with open(run_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Summarize instability-diagnostic runs from metrics and eval histories.")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    summaries = []
    for child in sorted(run_root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "run_metadata.json").exists():
            continue
        summaries.append(summarize_case(child))

    aggregate = {"runs": summaries}
    aggregate_path = run_root / "aggregate_summary.json"
    with open(aggregate_path, "w") as handle:
        json.dump(aggregate, handle, indent=2)

    print(json.dumps(aggregate, indent=2))
    print(f"Wrote aggregate summary to {aggregate_path}")


if __name__ == "__main__":
    main()
