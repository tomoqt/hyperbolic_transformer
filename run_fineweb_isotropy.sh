#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
COMPILE="${COMPILE:-False}"
MAX_ITERS="${MAX_ITERS:-1500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
PROBE_LAYERS="${PROBE_LAYERS:-all}"
ANALYSIS_BATCHES="${ANALYSIS_BATCHES:-4}"
ANALYSIS_BATCH_SIZE="${ANALYSIS_BATCH_SIZE:-4}"
ANALYSIS_MAX_TOKENS="${ANALYSIS_MAX_TOKENS:-4096}"
DATA_CHUNKS="${DATA_CHUNKS:-1}"
RUN_ROOT="${RUN_ROOT:-runs/fineweb_isotropy}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis_out/fineweb_isotropy}"
LEARNING_RATE="${LEARNING_RATE:-}"
USE_MUON="${USE_MUON:-False}"
MUON_LR_RATIO="${MUON_LR_RATIO:-100}"
MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
MUON_NESTEROV="${MUON_NESTEROV:-True}"
MUON_NS_STEPS="${MUON_NS_STEPS:-5}"

mkdir -p "$RUN_ROOT" "$ANALYSIS_ROOT"

prepare_data() {
    if [[ -f data/fineweb/train.bin && -f data/fineweb/val.bin && -f data/fineweb/meta.pkl ]]; then
        echo "==> Fineweb data already present"
        return
    fi

    echo "==> Preparing Fineweb data with $DATA_CHUNKS training chunk(s)"
    "$PYTHON_BIN" data/fineweb/prepare.py "$DATA_CHUNKS"
}

scale_muon_lr() {
    "$PYTHON_BIN" - <<'PY' "$1" "$2"
import sys

learning_rate = float(sys.argv[1])
ratio = float(sys.argv[2])
print(f"{learning_rate * ratio:.12g}")
PY
}

run_train() {
    local model_kind="$1"
    local use_baseline="$2"
    local out_dir="$RUN_ROOT/$model_kind"
    local log_file="$out_dir/train.log"
    local muon_lr=""
    local train_args=(
        --use_baseline_model="$use_baseline"
        --use_muon="$USE_MUON"
        --device="$DEVICE"
        --compile="$COMPILE"
        --wandb_log=False
        --max_iters="$MAX_ITERS"
        --lr_decay_iters="$MAX_ITERS"
        --eval_interval="$EVAL_INTERVAL"
        --log_interval="$LOG_INTERVAL"
        --always_save_checkpoint=True
        --out_dir="$out_dir"
        --muon_momentum="$MUON_MOMENTUM"
        --muon_nesterov="$MUON_NESTEROV"
        --muon_ns_steps="$MUON_NS_STEPS"
    )

    mkdir -p "$out_dir"
    echo "==> Training $model_kind model"
    if [[ -n "$LEARNING_RATE" ]]; then
        train_args+=(--learning_rate="$LEARNING_RATE")
    fi
    if [[ "$USE_MUON" == "True" && -n "$LEARNING_RATE" ]]; then
        muon_lr="$(scale_muon_lr "$LEARNING_RATE" "$MUON_LR_RATIO")"
        train_args+=(--muon_lr="$muon_lr")
    fi
    "$PYTHON_BIN" train.py config/train_fineweb_small.py "${train_args[@]}" | tee "$log_file"

    if [[ ! -f "$out_dir/ckpt.pt" ]]; then
        echo "Missing checkpoint for $model_kind at $out_dir/ckpt.pt" >&2
        exit 1
    fi
}

run_analysis() {
    local model_kind="$1"
    local out_dir="$RUN_ROOT/$model_kind"
    local output_name="${model_kind}_isotropy.json"

    echo "==> Analyzing $model_kind checkpoint"
    "$PYTHON_BIN" analyze_representations.py \
        --checkpoint "$out_dir/ckpt.pt" \
        --dataset fineweb \
        --split val \
        --device "$DEVICE" \
        --probe_layers "$PROBE_LAYERS" \
        --num_batches "$ANALYSIS_BATCHES" \
        --batch_size "$ANALYSIS_BATCH_SIZE" \
        --max_tokens "$ANALYSIS_MAX_TOKENS" \
        --output_dir "$ANALYSIS_ROOT" \
        --output_name "$output_name"
}

prepare_data

run_train baseline True
run_train hyperbolic False

run_analysis baseline
run_analysis hyperbolic

"$PYTHON_BIN" - <<'PY' "$ANALYSIS_ROOT/baseline_isotropy.json" "$ANALYSIS_ROOT/hyperbolic_isotropy.json" "$ANALYSIS_ROOT/summary.json"
import json
import sys

baseline_path, hyperbolic_path, summary_path = sys.argv[1:4]

with open(baseline_path) as f:
    baseline = json.load(f)
with open(hyperbolic_path) as f:
    hyperbolic = json.load(f)

common_layers = sorted(set(baseline["layers"]) & set(hyperbolic["layers"]), key=lambda x: int(x))
layer_deltas = {}
for layer in common_layers:
    base = baseline["layers"][layer]
    hyp = hyperbolic["layers"][layer]
    layer_deltas[layer] = {
        "normalized_spectral_entropy_delta": hyp["normalized_spectral_entropy"] - base["normalized_spectral_entropy"],
        "effective_rank_delta": hyp["effective_rank"] - base["effective_rank"],
        "participation_ratio_delta": hyp["participation_ratio"] - base["participation_ratio"],
        "top1_share_delta": hyp["top1_share"] - base["top1_share"],
        "condition_number_delta": hyp["condition_number"] - base["condition_number"],
    }

summary = {
    "baseline_report": baseline_path,
    "hyperbolic_report": hyperbolic_path,
    "common_layers": common_layers,
    "layer_deltas": layer_deltas,
    "last_layer": common_layers[-1] if common_layers else None,
}

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print(f"Wrote summary to {summary_path}")
PY
