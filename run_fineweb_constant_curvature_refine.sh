#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
COMPILE="${COMPILE:-False}"
RUN_ROOT="${RUN_ROOT:-runs/fineweb_constant_curvature_refine}"
BASE_CONFIG="${BASE_CONFIG:-config/train_fineweb_small.py}"
DATA_CHUNKS="${DATA_CHUNKS:-1}"
MAX_ITERS="${MAX_ITERS:-1500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
CURVATURE_GRID="${CURVATURE_GRID:-0.03 0.05 0.07 0.1 0.14 0.2 0.3}"
BASE_BLOCK_SIZE="${BASE_BLOCK_SIZE:-512}"
BASE_N_LAYER="${BASE_N_LAYER:-6}"
BASE_N_HEAD="${BASE_N_HEAD:-6}"
BASE_N_EMBD="${BASE_N_EMBD:-384}"
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

sanitize_tag() {
    echo "$1" | tr ' .:/' 'p___'
}

scale_muon_lr() {
    "$PYTHON_BIN" - <<'PY' "$1" "$2"
import sys

learning_rate = float(sys.argv[1])
ratio = float(sys.argv[2])
print(f"{learning_rate * ratio:.12g}")
PY
}

write_metadata() {
    local metadata_path="$1"
    local family="$2"
    local sweep_parameter_name="$3"
    local sweep_parameter_value="$4"
    local sweep_parameter_sort="$5"
    local sweep_parameter_display="$6"
    local model_kind="$7"
    local use_baseline="$8"
    local block_size="$9"
    local n_layer="${10}"
    local n_head="${11}"
    local n_embd="${12}"
    local curvature="${13}"
    local muon_lr="${14}"

    "$PYTHON_BIN" - <<'PY' "$metadata_path" "$family" "$sweep_parameter_name" "$sweep_parameter_value" "$sweep_parameter_sort" "$sweep_parameter_display" "$model_kind" "$use_baseline" "$block_size" "$n_layer" "$n_head" "$n_embd" "$curvature" "$LEARNING_RATE" "$USE_MUON" "$muon_lr" "$MUON_LR_RATIO" "$MAX_ITERS" "$EVAL_INTERVAL" "$LOG_INTERVAL"
import json
import sys

(
    metadata_path,
    family,
    sweep_parameter_name,
    sweep_parameter_value,
    sweep_parameter_sort,
    sweep_parameter_display,
    model_kind,
    use_baseline,
    block_size,
    n_layer,
    n_head,
    n_embd,
    curvature,
    learning_rate,
    use_muon,
    muon_lr,
    muon_lr_ratio,
    max_iters,
    eval_interval,
    log_interval,
) = sys.argv[1:21]

def parse_sort_value(raw):
    if raw.startswith("[") and raw.endswith("]"):
        return [int(part) for part in raw[1:-1].split(",") if part]
    if raw.startswith("(") and raw.endswith(")"):
        return [int(part) for part in raw[1:-1].split(",") if part]
    try:
        if "." in raw or "e" in raw.lower():
            return float(raw)
        return int(raw)
    except ValueError:
        return raw

metadata = {
    "family": family,
    "sweep_parameter_name": sweep_parameter_name,
    "sweep_parameter_value": sweep_parameter_value,
    "sweep_parameter_sort": parse_sort_value(sweep_parameter_sort),
    "sweep_parameter_display": sweep_parameter_display,
    "model_kind": model_kind,
    "use_baseline_model": use_baseline.lower() == "true",
    "optimizer": "muon" if use_muon.lower() == "true" else "adamw",
    "use_muon": use_muon.lower() == "true",
    "learning_rate": float(learning_rate),
    "muon_lr": float(muon_lr),
    "muon_lr_ratio": float(muon_lr_ratio),
    "max_iters": int(max_iters),
    "eval_interval": int(eval_interval),
    "log_interval": int(log_interval),
    "block_size": int(block_size),
    "n_layer": int(n_layer),
    "n_head": int(n_head),
    "n_embd": int(n_embd),
    "curvature_mode": "fixed",
    "curvature": float(curvature),
    "dynamic_curvature": False,
    "per_head_curvature": False,
    "use_embedding_curvature": False,
}

with open(metadata_path, "w") as handle:
    json.dump(metadata, handle, indent=2)
PY
}

run_one() {
    local family="$1"
    local sweep_parameter_name="$2"
    local sweep_parameter_value="$3"
    local sweep_parameter_sort="$4"
    local sweep_parameter_display="$5"
    local model_kind="$6"
    local use_baseline="$7"
    local block_size="$8"
    local n_layer="$9"
    local n_head="${10}"
    local n_embd="${11}"
    local curvature="${12}"
    local run_root="$RUN_ROOT/$family"
    local tag
    local out_dir
    local log_file
    local metadata_file
    local muon_lr

    tag="$(sanitize_tag "${model_kind}_${sweep_parameter_display}")"
    out_dir="$run_root/$tag"
    log_file="$out_dir/train.log"
    metadata_file="$out_dir/run_metadata.json"
    muon_lr="$(scale_muon_lr "$LEARNING_RATE" "$MUON_LR_RATIO")"

    mkdir -p "$out_dir"
    write_metadata \
        "$metadata_file" \
        "$family" \
        "$sweep_parameter_name" \
        "$sweep_parameter_value" \
        "$sweep_parameter_sort" \
        "$sweep_parameter_display" \
        "$model_kind" \
        "$use_baseline" \
        "$block_size" \
        "$n_layer" \
        "$n_head" \
        "$n_embd" \
        "$curvature" \
        "$muon_lr"

    echo "==> [$family] $model_kind @ $sweep_parameter_display"
    "$PYTHON_BIN" train.py "$BASE_CONFIG" \
        --use_baseline_model="$use_baseline" \
        --use_muon="$USE_MUON" \
        --device="$DEVICE" \
        --compile="$COMPILE" \
        --wandb_log=False \
        --learning_rate="$LEARNING_RATE" \
        --curvature_mode=fixed \
        --curvature="$curvature" \
        --dynamic_curvature=False \
        --per_head_curvature=False \
        --use_embedding_curvature=False \
        --block_size="$block_size" \
        --n_layer="$n_layer" \
        --n_head="$n_head" \
        --n_embd="$n_embd" \
        --muon_lr="$muon_lr" \
        --muon_momentum="$MUON_MOMENTUM" \
        --muon_nesterov="$MUON_NESTEROV" \
        --muon_ns_steps="$MUON_NS_STEPS" \
        --max_iters="$MAX_ITERS" \
        --lr_decay_iters="$MAX_ITERS" \
        --eval_interval="$EVAL_INTERVAL" \
        --log_interval="$LOG_INTERVAL" \
        --always_save_checkpoint=True \
        --out_dir="$out_dir" | tee "$log_file"
}

prepare_data

echo "==> Running curvature refinement sweep"
for curvature in $CURVATURE_GRID; do
    for model_kind in baseline hyperbolic; do
        if [[ "$model_kind" == "baseline" ]]; then
            use_baseline=True
        else
            use_baseline=False
        fi
        run_one \
            "curvature_refine" \
            "curvature" \
            "$curvature" \
            "$curvature" \
            "$curvature" \
            "$model_kind" \
            "$use_baseline" \
            "$BASE_BLOCK_SIZE" \
            "$BASE_N_LAYER" \
            "$BASE_N_HEAD" \
            "$BASE_N_EMBD" \
            "$curvature"
    done
done

"$PYTHON_BIN" summarize_fixed_curvature_sweep.py --run-root "$RUN_ROOT"
"$PYTHON_BIN" plot_fixed_curvature_sweep.py --run-root "$RUN_ROOT"
