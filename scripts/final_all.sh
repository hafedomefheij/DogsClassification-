#!/usr/bin/env bash
# The locked test evaluation of one experiment. Run it only after `scripts/freeze.py --experiment <exp>`
# and after the freeze is recorded in research_log/research_log.md. Every step below refuses to run
# without the freeze. The comparisons are the ones fixed in docs/design.md §8-9.
#
#   bash scripts/final_all.sh tsinghua_main  > $WORK/logs/final_tsinghua_main.log 2>&1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python}"
EXP="$1"
IMAGES="${TSINGHUA_IMAGES:-/content/data/raw/tsinghua_dogs/low-resolution.zip}"

join() { local IFS=,; echo "$*"; }
P=(rr_rival rr_rival_soft rr_pairwise)
pairs=()
for a in "${P[@]}"; do for b in rr_class rr_center rr_random; do pairs+=("$a:$b"); done; done   # H1
pairs+=(rr_pairwise:rr_rival rr_rival_soft:rr_rival)                                          # H4, soft
pairs+=(rr_random:rr_center rr_center:rr_class)                                               # H3, centring
for a in "${P[@]}"; do pairs+=("$a:classattn_all"); done                                      # H5
for a in "${P[@]}" rr_class; do pairs+=("$a:linear"); done                                    # H2, gains
pairs+=(rr_rival:attnprobe rr_pairwise:attnprobe)
MAIN=$(join "${pairs[@]}")
VIEWCMP=rr_rival:linear,rr_rival_soft:linear,rr_pairwise:linear,rr_class:linear,rr_rival:rr_class,rr_rival_soft:rr_class,rr_pairwise:rr_class
AUTO=blur,noise,jpeg,lowres,occlude

step() { echo "[$(date -u +%H:%M:%S)] $*"; "$@"; }
fe() { step "$PY" -I "$ROOT/scripts/final_eval.py" --experiment "$EXP" "$@"; }

case "$EXP" in
  tsinghua_main)
    fe --views orig --compare "$MAIN"
    fe --views "$AUTO" --compare "$VIEWCMP"
    fe --views head_occlude,bg_gray,bg_blur,crop_body,crop_head --compare "$VIEWCMP"   # annotation-assisted
    step "$PY" -I "$ROOT/scripts/subset_eval.py" --experiment "$EXP" --view orig --compare "$MAIN"
    step "$PY" -I "$ROOT/scripts/transfer_eval.py" --experiment "$EXP"
    step "$PY" -I "$ROOT/scripts/explain.py" --experiment "$EXP" --images "$IMAGES"
    ;;
  stanford_main)
    fe --views orig --compare "$MAIN"
    fe --views "$AUTO" --compare "$VIEWCMP"
    fe --views bg_gray,bg_blur,crop_body --compare "$VIEWCMP"                          # annotation-assisted
    ;;
  tsinghua_ablate)
    fe --views orig --compare rr_rival_k2:rr_class_k2,rr_rival_k3:rr_class_k3,rr_rival_k10:rr_class_k10,rr_pairwise_k3:rr_rival_k3,rr_pairwise_k10:rr_rival_k10
    ;;
  tsinghua_openset)
    step "$PY" -I "$ROOT/scripts/ood_eval.py" --experiment "$EXP"
    ;;
  tsinghua_resnet50|tsinghua_convnext_tiny|tsinghua_clip_vitb16)
    fe --views orig --compare rr_rival:rr_class,rr_pairwise:rr_class,rr_pairwise:rr_rival,rr_rival:linear,rr_class:linear
    ;;
  tsinghua_tok32|tsinghua_tok64)
    fe --views orig --compare rr_rival:rr_class,rr_rival:attnprobe,rr_class:attnprobe
    ;;
  *)
    echo "no evaluation plan for $EXP" >&2; exit 2 ;;
esac
echo "[$(date -u +%H:%M:%S)] FINAL DONE $EXP"
