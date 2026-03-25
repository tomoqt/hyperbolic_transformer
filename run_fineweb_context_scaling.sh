#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
COMPILE="${COMPILE:-False}"
MAX_ITERS="${MAX_ITERS:-1500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
DATA_CHUNKS="${DATA_CHUNKS:-1}"
RUN_ROOT="${RUN_ROOT:-runs/fineweb_context_scaling}"
BLOCK_SIZES="${BLOCK_SIZES:-512 768 1024}"
MODEL_KINDS="${MODEL_KINDS:-baseline mixed_curvature}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
CURVATURE="${CURVATURE:-0.1}"
USE_MUON="${USE_MUON:-False}"
MUON_LR_RATIO="${MUON_LR_RATIO:-1}"
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

sanitize_scale() {
    echo "$1" | tr '.-' '__'
}

context_grad_accum() {
    case "$1" in
        512) echo 8 ;;
        768) echo 6 ;;
        1024) echo 4 ;;
        *) echo 8 ;;
    esac
}

write_metadata() {
    local metadata_path="$1"
    local model_kind="$2"
    local use_baseline="$3"
    local block_size="$4"
    local grad_accum="$5"
    local scale_order="$6"
    local scale_label="$7"

    "$PYTHON_BIN" - <<'PY' "$metadata_path" "$model_kind" "$use_baseline" "$block_size" "$grad_accum" "$scale_order" "$scale_label" "$RUN_ROOT" "$LEARNING_RATE" "$CURVATURE" "$USE_MUON" "$MUON_LR_RATIO" "$MUON_MOMENTUM" "$MUON_NESTEROV" "$MUON_NS_STEPS" "$MAX_ITERS" "$EVAL_INTERVAL" "$LOG_INTERVAL" "$DEVICE" "$COMPILE"
import json
import sys

(
    metadata_path,
    model_kind,
    use_baseline,
    block_size,
    grad_accum,
    scale_order,
    scale_label,
    run_root,
    learning_rate,
    curvature,
    use_muon,
    muon_lr_ratio,
    muon_momentum,
    muon_nesterov,
    muon_ns_steps,
    max_iters,
    eval_interval,
    log_interval,
    device,
    compile_enabled,
) = sys.argv[1:]

block_size_int = int(block_size)
grad_accum_int = int(grad_accum)
tokens_per_iter = block_size_int * grad_accum_int
metadata = {
    "sweep_kind": "context",
    "scale_axis": "context_length",
    "scale_order": int(scale_order),
    "scale_label": scale_label,
    "scale_value": block_size_int,
    "model_kind": model_kind,
    "geometry_kind": "baseline" if use_baseline.lower() == "true" else "mixed_curvature",
    "use_baseline_model": use_baseline.lower() == "true",
    "learning_rate": float(learning_rate),
    "curvature": float(curvature),
    "dynamic_curvature": False,
    "per_head_curvature": False,
    "use_embedding_curvature": False,
    "use_muon": use_muon.lower() == "true",
    "muon_lr_ratio": float(muon_lr_ratio),
    "muon_momentum": float(muon_momentum),
    "muon_nesterov": muon_nesterov.lower() == "true",
    "muon_ns_steps": int(muon_ns_steps),
    "max_iters": int(max_iters),
    "eval_interval": int(eval_interval),
    "log_interval": int(log_interval),
    "device": device,
    "compile": compile_enabled.lower() == "true",
    "block_size": block_size_int,
    "batch_size": 1,
    "gradient_accumulation_steps": grad_accum_int,
    "tokens_per_iter": tokens_per_iter,
    "total_token_budget": tokens_per_iter * int(max_iters),
    "run_root": run_root,
}

with open(metadata_path, "w") as handle:
    json.dump(metadata, handle, indent=2)
PY
}

run_one() {
    local model_kind="$1"
    local use_baseline="$2"
    local block_size="$3"
    local grad_accum="$4"
    local scale_order="$5"
    local scale_label="$6"
    local scale_tag
    local out_dir
    local log_file
    local metadata_file

    scale_tag="ctx_${scale_label}"
    out_dir="$RUN_ROOT/${model_kind}_${scale_tag}"
    log_file="$out_dir/train.log"
    metadata_file="$out_dir/run_metadata.json"

    mkdir -p "$out_dir"
    write_metadata "$metadata_file" "$model_kind" "$use_baseline" "$block_size" "$grad_accum" "$scale_order" "$scale_label"

    echo "==> Running $model_kind at block_size=$block_size grad_accum=$grad_accum"

    args=(
        "$PYTHON_BIN" train.py config/train_fineweb_small.py
        --use_baseline_model="$use_baseline"
        --use_muon="$USE_MUON"
        --device="$DEVICE"
        --compile="$COMPILE"
        --wandb_log=False
        --learning_rate="$LEARNING_RATE"
        --muon_lr="$(printf '%s\n' "$LEARNING_RATE" | awk -v ratio="$MUON_LR_RATIO" '{printf "%.12g", $1 * ratio}')"
        --muon_momentum="$MUON_MOMENTUM"
        --muon_nesterov="$MUON_NESTEROV"
        --muon_ns_steps="$MUON_NS_STEPS"
        --batch_size=1
        --block_size="$block_size"
        --gradient_accumulation_steps="$grad_accum"
        --max_iters="$MAX_ITERS"
        --lr_decay_iters="$MAX_ITERS"
        --eval_interval="$EVAL_INTERVAL"
        --log_interval="$LOG_INTERVAL"
        --always_save_checkpoint=True
        --out_dir="$out_dir"
    )

    if [[ "$use_baseline" == "False" ]]; then
        args+=(
            --curvature_mode=fixed
            --curvature="$CURVATURE"
            --dynamic_curvature=False
            --per_head_curvature=False
            --use_embedding_curvature=False
        )
    fi

    if ! "${args[@]}" | tee "$log_file"; then
        echo "❌ Failed: $model_kind at block_size=$block_size"
    fi
}

prepare_data

scale_order=0
for block_size in $BLOCK_SIZES; do
    grad_accum="$(context_grad_accum "$block_size")"
    for model_kind in $MODEL_KINDS; do
        if [[ "$model_kind" == "baseline" ]]; then
            use_baseline=True
        else
            use_baseline=False
        fi
        run_one "$model_kind" "$use_baseline" "$block_size" "$grad_accum" "$scale_order" "$block_size"
    done
    scale_order=$((scale_order + 1))
done

"$PYTHON_BIN" summarize_scaling_sweep.py --run-root "$RUN_ROOT"
"$PYTHON_BIN" plot_scaling_sweep.py --run-root "$RUN_ROOT"
