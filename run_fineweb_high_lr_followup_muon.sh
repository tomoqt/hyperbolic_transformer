#!/usr/bin/env bash
set -euo pipefail

export USE_MUON="${USE_MUON:-True}"
export MUON_LR_RATIO="${MUON_LR_RATIO:-100}"
export RUN_ROOT="${RUN_ROOT:-runs/fineweb_high_lr_followup_muon}"

./run_fineweb_high_lr_followup.sh
