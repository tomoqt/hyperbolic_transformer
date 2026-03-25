#!/usr/bin/env bash
set -uo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
COMPILE="${COMPILE:-False}"
MAX_ITERS="${MAX_ITERS:-1500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
RUN_ROOT="${RUN_ROOT:-runs/shakespeare_lr_sweep}"
LEARNING_RATES="${LEARNING_RATES:-1e-3 5e-4 2e-4 1e-4}"
USE_MUON="${USE_MUON:-False}"
MUON_LR_RATIO="${MUON_LR_RATIO:-10}"
MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
MUON_NESTEROV="${MUON_NESTEROV:-True}"
MUON_NS_STEPS="${MUON_NS_STEPS:-5}"

mkdir -p "$RUN_ROOT"

sanitize_lr() {
    echo "$1" | tr '.-' '__'
}

scale_muon_lr() {
    "$PYTHON_BIN" - <<'PY' "$1" "$2"
import sys

learning_rate = float(sys.argv[1])
ratio = float(sys.argv[2])
print(f"{learning_rate * ratio:.12g}")
PY
}

run_one() {
    local model_kind="$1"
    local use_baseline="$2"
    local lr="$3"
    local lr_tag
    local out_dir
    local log_file
    local summary_file
    local metadata_file
    local runtime=0
    local success=0
    local muon_lr

    lr_tag="$(sanitize_lr "$lr")"
    out_dir="$RUN_ROOT/${model_kind}_lr_${lr_tag}"
    log_file="$out_dir/train.log"
    summary_file="$out_dir/summary.json"
    metadata_file="$out_dir/run_metadata.json"
    muon_lr="$(scale_muon_lr "$lr" "$MUON_LR_RATIO")"

    mkdir -p "$out_dir"
    echo "==> Running $model_kind at lr=$lr"

    "$PYTHON_BIN" - <<'PY' "$metadata_file" "$model_kind" "$lr" "$USE_MUON" "$muon_lr" "$MUON_LR_RATIO"
import json
import sys

metadata_path, model_kind, learning_rate, use_muon, muon_lr, muon_lr_ratio = sys.argv[1:7]

metadata = {
    "model_kind": model_kind,
    "learning_rate": learning_rate,
    "use_muon": use_muon.lower() == "true",
    "optimizer": "muon" if use_muon.lower() == "true" else "adamw",
    "muon_lr": float(muon_lr),
    "muon_lr_ratio": float(muon_lr_ratio),
}

with open(metadata_path, "w") as handle:
    json.dump(metadata, handle, indent=2)
PY

    SECONDS=0
    if "$PYTHON_BIN" train.py config/train_shakespeare_char.py \
        --use_baseline_model="$use_baseline" \
        --use_muon="$USE_MUON" \
        --device="$DEVICE" \
        --compile="$COMPILE" \
        --wandb_log=False \
        --learning_rate="$lr" \
        --muon_lr="$muon_lr" \
        --muon_momentum="$MUON_MOMENTUM" \
        --muon_nesterov="$MUON_NESTEROV" \
        --muon_ns_steps="$MUON_NS_STEPS" \
        --max_iters="$MAX_ITERS" \
        --lr_decay_iters="$MAX_ITERS" \
        --eval_interval="$EVAL_INTERVAL" \
        --log_interval="$LOG_INTERVAL" \
        --always_save_checkpoint=True \
        --out_dir="$out_dir" | tee "$log_file"; then
        success=1
    fi
    runtime=$SECONDS

    "$PYTHON_BIN" - <<'PY' "$log_file" "$summary_file" "$model_kind" "$lr" "$runtime" "$success" "$out_dir/ckpt.pt" "$USE_MUON" "$muon_lr" "$MUON_LR_RATIO"
import json
import os
import re
import sys

import torch

log_path, summary_path, model_kind, learning_rate, runtime_seconds, success_flag, checkpoint_path, use_muon, muon_lr, muon_lr_ratio = sys.argv[1:11]
runtime_seconds = int(runtime_seconds)
success = bool(int(success_flag))
step_pattern = re.compile(r"step (\d+): train loss ([0-9.]+), val loss ([0-9.]+)")

history = []
with open(log_path) as f:
    for line in f:
        match = step_pattern.search(line)
        if match:
            history.append(
                {
                    "step": int(match.group(1)),
                    "train_loss": float(match.group(2)),
                    "val_loss": float(match.group(3)),
                }
            )

best = min(history, key=lambda item: item["val_loss"]) if history else None
checkpoint = None
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def to_jsonable(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value

summary = {
    "model_kind": model_kind,
    "learning_rate": learning_rate,
    "use_muon": use_muon.lower() == "true",
    "optimizer": "muon" if use_muon.lower() == "true" else "adamw",
    "muon_lr": float(muon_lr),
    "muon_lr_ratio": float(muon_lr_ratio),
    "runtime_seconds": runtime_seconds,
    "success": success,
    "num_evals": len(history),
    "best_eval": best,
    "checkpoint_path": checkpoint_path if os.path.exists(checkpoint_path) else None,
    "checkpoint_iter_num": to_jsonable(checkpoint.get("iter_num")) if checkpoint else None,
    "checkpoint_best_val_loss": to_jsonable(checkpoint.get("best_val_loss")) if checkpoint else None,
}

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print(f"Wrote summary to {summary_path}")
PY
}

for model_kind in baseline hyperbolic; do
    if [[ "$model_kind" == "baseline" ]]; then
        use_baseline=True
    else
        use_baseline=False
    fi
    for lr in $LEARNING_RATES; do
        run_one "$model_kind" "$use_baseline" "$lr"
    done
done

"$PYTHON_BIN" summarize_lr_sweep.py --run-root "$RUN_ROOT"
