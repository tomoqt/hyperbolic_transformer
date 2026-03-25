#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


STEP_RE = re.compile(r"step\s+(\d+):\s+train loss\s+([0-9.]+),\s+val loss\s+([0-9.]+)")
ITER_RE = re.compile(r"iter\s+(\d+):\s+loss\s+([0-9.]+),\s+time\s+([0-9.]+)ms")


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
            if line:
                records.append(json.loads(line))
    return records


def is_finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def parse_history_from_metrics(metrics):
    history = []
    for record in metrics:
        if record.get("event") != "eval":
            continue
        train_loss = record.get("train_loss")
        val_loss = record.get("val_loss")
        if not (is_finite_number(train_loss) and is_finite_number(val_loss)):
            continue
        history.append(
            {
                "step": int(record.get("iter", 0)),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "lr": record.get("lr"),
                "mfu": record.get("mfu"),
            }
        )
    return history


def summarize_step_times(metrics, tokens_per_iter):
    train_steps = [record for record in metrics if record.get("event") == "train_step" and is_finite_number(record.get("time_ms"))]
    if not train_steps:
        return {
            "mean_train_step_ms": None,
            "median_train_step_ms": None,
            "mean_train_tokens_per_sec": None,
            "median_train_tokens_per_sec": None,
            "train_step_count": 0,
        }

    times_ms = [float(record["time_ms"]) for record in train_steps]
    times_ms_sorted = sorted(times_ms)
    mid = len(times_ms_sorted) // 2
    if len(times_ms_sorted) % 2:
        median_ms = times_ms_sorted[mid]
    else:
        median_ms = 0.5 * (times_ms_sorted[mid - 1] + times_ms_sorted[mid])

    mean_ms = sum(times_ms) / len(times_ms)
    if isinstance(tokens_per_iter, str):
        tokens_per_iter = float(tokens_per_iter)
    median_tokens_per_sec = None if not tokens_per_iter else (float(tokens_per_iter) / median_ms) * 1000.0
    mean_tokens_per_sec = None if not tokens_per_iter else (float(tokens_per_iter) / mean_ms) * 1000.0
    return {
        "mean_train_step_ms": mean_ms,
        "median_train_step_ms": median_ms,
        "mean_train_tokens_per_sec": mean_tokens_per_sec,
        "median_train_tokens_per_sec": median_tokens_per_sec,
        "train_step_count": len(train_steps),
    }


def pick_best_eval(history):
    if not history:
        return None
    return min(history, key=lambda record: record["val_loss"])


def summarize_run(run_dir: Path):
    metadata = load_json(run_dir / "run_metadata.json")
    manifest = load_json(run_dir / "run_manifest.json")
    metrics = load_jsonl(run_dir / "metrics.jsonl")
    eval_history = load_json(run_dir / "eval_history.json")

    if not eval_history:
        eval_history = parse_history_from_metrics(metrics)

    with open(run_dir / "history.json", "w") as handle:
        json.dump(eval_history, handle, indent=2)

    best_eval = pick_best_eval(eval_history)
    final_eval = eval_history[-1] if eval_history else None
    run_manifest = manifest.get("run", {})
    model_manifest = manifest.get("model", {})
    optimizer_manifest = manifest.get("optimizer", {})
    config_manifest = manifest.get("config", {})

    tokens_per_iter = run_manifest.get("tokens_per_iter") or metadata.get("tokens_per_iter")
    total_token_budget = run_manifest.get("total_token_budget") or metadata.get("total_token_budget")
    time_stats = summarize_step_times(metrics, tokens_per_iter)

    summary = {
        "run_dir": str(run_dir),
        "sweep_kind": metadata.get("sweep_kind"),
        "scale_axis": metadata.get("scale_axis"),
        "scale_order": metadata.get("scale_order"),
        "scale_label": metadata.get("scale_label"),
        "scale_value": metadata.get("scale_value"),
        "model_kind": metadata.get("model_kind"),
        "geometry_kind": metadata.get("geometry_kind"),
        "use_baseline_model": metadata.get("use_baseline_model"),
        "learning_rate": metadata.get("learning_rate"),
        "curvature_mode": metadata.get("curvature_mode"),
        "curvature": metadata.get("curvature"),
        "dynamic_curvature": metadata.get("dynamic_curvature"),
        "per_head_curvature": metadata.get("per_head_curvature"),
        "use_embedding_curvature": metadata.get("use_embedding_curvature"),
        "block_size": metadata.get("block_size"),
        "batch_size": metadata.get("batch_size"),
        "gradient_accumulation_steps": metadata.get("gradient_accumulation_steps"),
        "n_layer": metadata.get("n_layer"),
        "n_embd": metadata.get("n_embd"),
        "n_head": metadata.get("n_head"),
        "tokens_per_iter": tokens_per_iter,
        "total_token_budget": total_token_budget,
        "max_iters": metadata.get("max_iters"),
        "eval_interval": metadata.get("eval_interval"),
        "log_interval": metadata.get("log_interval"),
        "device": metadata.get("device"),
        "compile": metadata.get("compile"),
        "use_muon": metadata.get("use_muon"),
        "muon_lr_ratio": metadata.get("muon_lr_ratio"),
        "muon_momentum": metadata.get("muon_momentum"),
        "muon_nesterov": metadata.get("muon_nesterov"),
        "muon_ns_steps": metadata.get("muon_ns_steps"),
        "success": (run_dir / "ckpt.pt").exists(),
        "num_eval_records": len(eval_history),
        "best_eval": best_eval,
        "final_eval": final_eval,
        "mean_train_step_ms": time_stats["mean_train_step_ms"],
        "median_train_step_ms": time_stats["median_train_step_ms"],
        "mean_train_tokens_per_sec": time_stats["mean_train_tokens_per_sec"],
        "median_train_tokens_per_sec": time_stats["median_train_tokens_per_sec"],
        "train_step_count": time_stats["train_step_count"],
        "manifest_path": str(run_dir / "run_manifest.json") if (run_dir / "run_manifest.json").exists() else None,
        "metrics_path": str(run_dir / "metrics.jsonl") if (run_dir / "metrics.jsonl").exists() else None,
        "eval_history_path": str(run_dir / "eval_history.json") if (run_dir / "eval_history.json").exists() else None,
        "history_path": str(run_dir / "history.json"),
        "checkpoint_path": str(run_dir / "ckpt.pt") if (run_dir / "ckpt.pt").exists() else None,
        "run_manifest": run_manifest,
        "model_manifest": model_manifest,
        "optimizer_manifest": optimizer_manifest,
        "config_manifest": config_manifest,
    }

    summary.update(metadata)

    with open(run_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def build_paired_comparisons(summaries):
    by_axis = defaultdict(dict)
    for summary in summaries:
        key = summary.get("scale_label")
        model_kind = summary.get("model_kind")
        by_axis[key][model_kind] = summary

    comparisons = []
    for scale_label, entries in sorted(
        by_axis.items(),
        key=lambda item: (
            item[1].get("baseline", {}).get("scale_order", 9999),
            str(item[0]),
        ),
    ):
        baseline = entries.get("baseline")
        mixed = entries.get("mixed_curvature")
        if not baseline or not mixed:
            continue

        baseline_best = baseline.get("best_eval") or {}
        mixed_best = mixed.get("best_eval") or {}
        baseline_time = baseline.get("mean_train_step_ms")
        mixed_time = mixed.get("mean_train_step_ms")
        comparisons.append(
            {
                "scale_label": scale_label,
                "scale_order": baseline.get("scale_order"),
                "scale_axis": baseline.get("scale_axis"),
                "baseline": {
                    "best_val_loss": baseline_best.get("val_loss"),
                    "mean_train_step_ms": baseline_time,
                    "mean_train_tokens_per_sec": baseline.get("mean_train_tokens_per_sec"),
                    "tokens_per_iter": baseline.get("tokens_per_iter"),
                },
                "mixed_curvature": {
                    "best_val_loss": mixed_best.get("val_loss"),
                    "mean_train_step_ms": mixed_time,
                    "mean_train_tokens_per_sec": mixed.get("mean_train_tokens_per_sec"),
                    "tokens_per_iter": mixed.get("tokens_per_iter"),
                },
                "deltas": {
                    "best_val_loss": None if None in (baseline_best.get("val_loss"), mixed_best.get("val_loss")) else mixed_best["val_loss"] - baseline_best["val_loss"],
                    "mean_train_step_ms": None if None in (baseline_time, mixed_time) else mixed_time - baseline_time,
                    "mean_train_step_ms_ratio": None if not baseline_time else mixed_time / baseline_time,
                    "mean_train_tokens_per_sec_ratio": None
                    if not baseline.get("mean_train_tokens_per_sec")
                    else mixed.get("mean_train_tokens_per_sec") / baseline.get("mean_train_tokens_per_sec")
                    if mixed.get("mean_train_tokens_per_sec")
                    else None,
                },
            }
        )
    return comparisons


def main():
    parser = argparse.ArgumentParser(description="Summarize a context/model scaling sweep.")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    summaries = []
    for child in sorted(run_root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "run_metadata.json").exists():
            continue
        summaries.append(summarize_run(child))

    paired_comparisons = build_paired_comparisons(summaries)
    aggregate = {
        "run_root": str(run_root),
        "sweep_kind": summaries[0]["sweep_kind"] if summaries else None,
        "scale_axis": summaries[0]["scale_axis"] if summaries else None,
        "run_count": len(summaries),
        "runs": summaries,
        "paired_comparisons": paired_comparisons,
    }

    aggregate_path = run_root / "aggregate_summary.json"
    with open(aggregate_path, "w") as handle:
        json.dump(aggregate, handle, indent=2)

    print(json.dumps(aggregate, indent=2))
    print(f"Wrote aggregate summary to {aggregate_path}")


if __name__ == "__main__":
    main()
