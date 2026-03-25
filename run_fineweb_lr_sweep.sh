#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
COMPILE="${COMPILE:-False}"
MAX_ITERS="${MAX_ITERS:-1500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
LEARNING_RATES="${LEARNING_RATES:-1e-3 5e-4 2e-4 1e-4}"
DATA_CHUNKS="${DATA_CHUNKS:-1}"
RUN_ROOT="${RUN_ROOT:-runs/fineweb_lr_sweep}"
MODEL_KINDS="${MODEL_KINDS:-baseline hyperbolic}"
USE_MUON="${USE_MUON:-False}"
MUON_LR_RATIO="${MUON_LR_RATIO:-100}"
MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
MUON_NESTEROV="${MUON_NESTEROV:-True}"
MUON_NS_STEPS="${MUON_NS_STEPS:-5}"

mkdir -p "$RUN_ROOT"

prepare_data() {
    if [[ -f data/fineweb/train.bin && -f data/fineweb/val.bin && -f data/fineweb/meta.pkl ]]; then
        echo "==> Fineweb data already present"
        return
    fi

    echo "==> Preparing Fineweb data with $DATA_CHUNKS training chunk(s)"
    "$PYTHON_BIN" data/fineweb/prepare.py "$DATA_CHUNKS"
}

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
    local metadata_file
    local muon_lr

    lr_tag="$(sanitize_lr "$lr")"
    out_dir="$RUN_ROOT/${model_kind}_lr_${lr_tag}"
    log_file="$out_dir/train.log"
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

    if "$PYTHON_BIN" train.py config/train_fineweb_small.py \
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
        true
    fi
}

prepare_data

for model_kind in $MODEL_KINDS; do
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
