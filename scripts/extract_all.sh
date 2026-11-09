#!/usr/bin/env bash
# Every feature cache one backbone needs, in order (each step is resumable, so a dropped
# Colab runtime just re-runs this script):
#   1. orig view of train, val and test; the token PCA is fitted here, on TRAINING images only
#   2. hflip view of train (training-time augmentation; val and test are never flipped)
#   3. test-time perturbations of the test split: the automatic ones first, then the
#      annotation-assisted ones (they use the ground-truth boxes and are reported as such)
#
#   bash scripts/extract_all.sh <dataset> <backbone> <images> [token_pca=256|none] [img_size=224]
#
# Environment: WORKERS (data-loader processes, default 2 = a Colab VM), BATCH (default 128),
# AUTO_VIEWS / ANN_VIEWS (comma lists; empty skips that step).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ds="$1"; bb="$2"; imgs="$3"; pca="${4:-256}"; size="${5:-224}"
PY="${PYTHON:-python}"
if [[ "$pca" == none ]]; then tok=(--no-tokens); else tok=(--token-pca "$pca"); fi

run() {
  echo "[$(date +%H:%M:%S)] extract $ds $bb $*"
  "$PY" -I "$ROOT/scripts/extract_features.py" --dataset "$ds" --backbone "$bb" --images "$imgs" \
    --img-size "$size" "${tok[@]}" --workers "${WORKERS:-2}" --batch-size "${BATCH:-128}" "$@"
}

run --splits train,val,test --views orig
run --splits train --views hflip
auto="${AUTO_VIEWS-blur,noise,jpeg,lowres,occlude}"
ann="${ANN_VIEWS-head_occlude,bg_gray,bg_blur,crop_body,crop_head}"
[[ -n "$auto" ]] && run --splits test --views "$auto"
[[ -n "$ann" ]] && run --splits test --views "$ann"
echo "[$(date +%H:%M:%S)] EXTRACT DONE $ds $bb"
