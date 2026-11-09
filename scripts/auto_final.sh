#!/usr/bin/env bash
# Unattended finishing of the secondary experiments (research log, "Automatic freeze and evaluation").
# For each queue job below, once scripts/queue.sh marks it done:
#   1. freeze the experiment (scripts/freeze.py), keeping its printout in $LOGS/freeze_<exp>.log;
#   2. run its locked test evaluation (scripts/final_all.sh), output in $LOGS/final_<exp>.log.
# One experiment at a time, in the order they finish. A failed job, a refused freeze or a failed evaluation is
# logged and skipped (never retried). Nothing here chooses anything: the evaluation plans are the ones fixed in
# scripts/final_all.sh before any of these experiments finished.
# After the open-set split job, its split is copied to $WORK/openset_backup. When every experiment is finished or
# skipped: the efficiency report (the GPU is quiet then), the digest, the paper tables and the figures.
#
#   source /content/work/env.sh
#   setsid nohup bash scripts/auto_final.sh > /content/work/logs/auto_final.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
WORK="${WORK:-/content/work}"
LOGS="${LOGS:-$WORK/logs}"
RES="${DOGBREED_RESULTS:-$WORK/results}"
IMAGES="${TSINGHUA_IMAGES:-/content/data/raw/tsinghua_dogs/low-resolution.zip}"
# queue/job:experiment
JOBS="${AUTO_FINAL_JOBS:-a/ablate:tsinghua_ablate a/openset:tsinghua_openset c/bb_resnet50:tsinghua_resnet50
      c/bb_convnext_tiny:tsinghua_convnext_tiny c/bb_clip_vitb16:tsinghua_clip_vitb16
      a/tok32:tsinghua_tok32 a/tok64:tsinghua_tok64}"
# AUTO_FINAL_REPORTS=efficiency: only the efficiency report at the end (a runtime that holds the new experiments
# but not the main results, such as the Kaggle session; the digest, tables and figures are then made where every
# result is, from the Drive mirror).
REPORTS="${AUTO_FINAL_REPORTS:-all}"
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
STATE="$LOGS/auto_final"
mkdir -p "$STATE"
say "watching: $(echo $JOBS)"

while true; do
  if [[ ! -e "$STATE/openset_split.saved" && -e "$LOGS/queue_a/openset_split.done" ]]; then
    if cp -r configs/openset "$WORK/openset_backup"; then
      touch "$STATE/openset_split.saved"; say "open-set split copied to $WORK/openset_backup"
    fi
  fi
  pending=0
  for jx in $JOBS; do
    q="${jx%%/*}"; rest="${jx#*/}"; job="${rest%%:*}"; exp="${rest#*:}"
    [[ -e "$STATE/$exp.done" || -e "$STATE/$exp.skipped" ]] && continue
    if [[ -e "$LOGS/queue_$q/$job.failed" ]]; then
      say "SKIP $exp: queue job $job failed (see $LOGS/$job.log)"; touch "$STATE/$exp.skipped"; continue
    fi
    if [[ ! -e "$LOGS/queue_$q/$job.done" ]]; then pending=1; continue; fi
    say "FREEZE $exp"
    if ! "$PY" -I scripts/freeze.py --experiment "$exp" > "$LOGS/freeze_$exp.log" 2>&1; then
      say "SKIP $exp: freeze refused (see $LOGS/freeze_$exp.log)"; touch "$STATE/$exp.skipped"; continue
    fi
    head -n 1 "$LOGS/freeze_$exp.log"
    say "FINAL $exp"
    if bash scripts/final_all.sh "$exp" > "$LOGS/final_$exp.log" 2>&1; then
      touch "$STATE/$exp.done"; say "FINAL DONE $exp"
    else
      touch "$STATE/$exp.skipped"; say "FINAL FAILED $exp (see $LOGS/final_$exp.log)"
    fi
  done
  (( pending )) || break
  sleep 60
done

say "every experiment finished or skipped; reports"
"$PY" -I scripts/efficiency_report.py --config configs/experiments/tsinghua_main.yaml > "$LOGS/efficiency.log" 2>&1
say "efficiency rc=$?"
if [[ "$REPORTS" == efficiency ]]; then say "AUTO FINAL END (efficiency only)"; exit 0; fi
"$PY" -I scripts/digest.py --out "$RES/digest" \
  --curves tsinghua_main/linear_s0,tsinghua_main/rr_class_s0,tsinghua_main/rr_rival_s0,tsinghua_main/rr_rival_soft_s0,tsinghua_main/rr_pairwise_s0 > "$LOGS/digest.log" 2>&1
say "digest rc=$?"
"$PY" -I scripts/make_tables.py --digest "$RES/digest" --out paper/generated > "$LOGS/tables.log" 2>&1
say "tables rc=$?"
"$PY" -I scripts/make_figures.py --gen paper/generated --out paper/figures --explain "$RES/tsinghua_main/explain/shown_maps.json" \
  --images "$IMAGES" --png > "$LOGS/figures.log" 2>&1
say "figures rc=$?"
say "AUTO FINAL END"
