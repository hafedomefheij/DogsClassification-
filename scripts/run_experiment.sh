#!/usr/bin/env bash
# Train every head of an experiment: learning-rate selection on seed 0 (validation top-1),
# then all seeds at the chosen rates. Never reads the test split (final_eval.py does that,
# once, after model selection is frozen).
#
#   bash scripts/run_experiment.sh configs/experiments/tsinghua_main.yaml [--only a,b]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cfg="$1"; shift
PY="${PYTHON:-python}"
echo "[$(date +%H:%M:%S)] lr selection $cfg"
"$PY" -I "$ROOT/scripts/run_heads.py" --config "$cfg" --select-lr "$@"
echo "[$(date +%H:%M:%S)] seeds $cfg"
"$PY" -I "$ROOT/scripts/run_heads.py" --config "$cfg" "$@"
echo "[$(date +%H:%M:%S)] TRAINING DONE $cfg"
