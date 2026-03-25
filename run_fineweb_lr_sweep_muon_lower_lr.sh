#!/usr/bin/env bash
set -euo pipefail

export USE_MUON="${USE_MUON:-True}"
export MUON_LR_RATIO="${MUON_LR_RATIO:-1}"
export LEARNING_RATES="${LEARNING_RATES:-2e-4 1.5e-4 1e-4}"
export RUN_ROOT="${RUN_ROOT:-runs/fineweb_lr_sweep_muon_lower_lr}"

./run_fineweb_lr_sweep.sh
