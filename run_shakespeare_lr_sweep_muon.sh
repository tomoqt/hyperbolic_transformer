#!/usr/bin/env bash
set -euo pipefail

export USE_MUON="${USE_MUON:-True}"
export MUON_LR_RATIO="${MUON_LR_RATIO:-3}"
export RUN_ROOT="${RUN_ROOT:-runs/shakespeare_lr_sweep_muon}"

./run_shakespeare_lr_sweep.sh
