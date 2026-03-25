#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
COMPILE="${COMPILE:-False}"
MAX_ITERS="${MAX_ITERS:-1500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
RUN_ROOT="${RUN_ROOT:-runs/shakespeare_muon_mobius_ablations}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
MUON_LR_RATIO="${MUON_LR_RATIO:-3}"
MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
MUON_NESTEROV="${MUON_NESTEROV:-True}"
MUON_NS_STEPS="${MUON_NS_STEPS:-5}"

mkdir -p "$RUN_ROOT"

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
    local case_name="$2"
    local residual_mode="$3"
    local transport_mode="$4"
    local project_points="$5"
    local muon_lr="$6"
    "$PYTHON_BIN" - <<'PY' "$metadata_path" "$case_name" "$LEARNING_RATE" "$residual_mode" "$transport_mode" "$project_points" "$MUON_LR_RATIO" "$muon_lr"
import json
import sys

(
    metadata_path,
    case_name,
    learning_rate,
    residual_mode,
    transport_mode,
    project_points,
    muon_lr_ratio,
    muon_lr,
) = sys.argv[1:9]

payload = {
    "case_name": case_name,
    "learning_rate": float(learning_rate),
    "dtype": "float16",
    "use_muon": True,
    "muon_unscale_grads": True,
    "hyperbolic_residual_mode": residual_mode,
    "hyperbolic_transport_mode": transport_mode,
    "project_hyperbolic_points": project_points.lower() == "true",
    "muon_lr_ratio": float(muon_lr_ratio),
    "muon_lr": float(muon_lr),
    "diagnostic_family": "mobius_ablations",
}

with open(metadata_path, "w") as handle:
    json.dump(payload, handle, indent=2)
PY
}

run_case() {
    local case_name="$1"
    local residual_mode="$2"
    local transport_mode="$3"
    local project_points="$4"
    local out_dir="$RUN_ROOT/$case_name"
    local log_file="$out_dir/train.log"
    local metadata_file="$out_dir/run_metadata.json"
    local muon_lr

    mkdir -p "$out_dir"
    muon_lr="$(scale_muon_lr "$LEARNING_RATE" "$MUON_LR_RATIO")"
    write_metadata "$metadata_file" "$case_name" "$residual_mode" "$transport_mode" "$project_points" "$muon_lr"

    echo "==> Running $case_name"
    "$PYTHON_BIN" train.py config/train_shakespeare_char.py \
        --use_baseline_model=False \
        --use_muon=True \
        --device="$DEVICE" \
        --compile="$COMPILE" \
        --dtype="float16" \
        --wandb_log=False \
        --learning_rate="$LEARNING_RATE" \
        --muon_lr="$muon_lr" \
        --muon_momentum="$MUON_MOMENTUM" \
        --muon_nesterov="$MUON_NESTEROV" \
        --muon_ns_steps="$MUON_NS_STEPS" \
        --muon_unscale_grads=True \
        --optimizer_debug=True \
        --hyperbolic_debug=True \
        --hyperbolic_residual_mode="$residual_mode" \
        --hyperbolic_transport_mode="$transport_mode" \
        --project_hyperbolic_points="$project_points" \
        --max_iters="$MAX_ITERS" \
        --lr_decay_iters="$MAX_ITERS" \
        --eval_interval="$EVAL_INTERVAL" \
        --log_interval="$LOG_INTERVAL" \
        --always_save_checkpoint=True \
        --out_dir="$out_dir" | tee "$log_file"
}

run_case "full_fp16_unscaled" "mobius" "logexp" "False"
run_case "no_mobius_fp16_unscaled" "euclidean" "logexp" "False"
run_case "projection_fp16_unscaled" "mobius" "logexp" "True"

"$PYTHON_BIN" summarize_instability_diagnostics.py --run-root "$RUN_ROOT"
"$PYTHON_BIN" plot_instability_diagnostics.py --run-root "$RUN_ROOT"
