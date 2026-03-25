#!/usr/bin/env bash
set -euo pipefail

export USE_MUON="${USE_MUON:-True}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export MUON_LR_RATIO="${MUON_LR_RATIO:-100}"
export RUN_ROOT="${RUN_ROOT:-runs/fineweb_isotropy_muon}"
export ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis_out/fineweb_isotropy_muon}"

./run_fineweb_isotropy.sh
