#!/usr/bin/env bash
# Stanford Dogs feature caches and the Tsinghua <-> Stanford overlap audit. Run after the
# Tsinghua caches of the same backbone exist (the transfer caches reuse the Tsinghua PCA).
#
#   bash scripts/pipeline_stanford.sh [stanford images dir] [tsinghua images] [backbone]
#
# 1. Stanford orig / hflip / test perturbations in Stanford's own PCA basis (pca256); the
#    annotation-assisted views are the body-box ones (Stanford has no head boxes);
# 2. Stanford test in the TSINGHUA PCA basis (pca256_tsinghua), for Tsinghua -> Stanford transfer;
# 3. scripts/cross_overlap.py: duplicates across the two datasets, the clean Stanford test list.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
simgs="${1:-/content/data/raw/stanford_dogs/Images}"
timgs="${2:-/content/data/raw/tsinghua_dogs/low-resolution.zip}"
bb="${3:-dinov2_vitb14_reg}"
PY="${PYTHON:-python}"
ANN_VIEWS="${ANN_VIEWS-bg_gray,bg_blur,crop_body}" bash "$ROOT/scripts/extract_all.sh" stanford "$bb" "$simgs" 256 224
echo "[$(date +%H:%M:%S)] stanford test in the tsinghua PCA basis"
"$PY" -I "$ROOT/scripts/extract_features.py" --dataset stanford --backbone "$bb" --images "$simgs" \
  --splits test --views orig --token-pca 256 --pca-from tsinghua --workers "${WORKERS:-2}"
echo "[$(date +%H:%M:%S)] cross-dataset overlap"
"$PY" -I "$ROOT/scripts/cross_overlap.py" --backbone "$bb" --feature-tag pca256 \
  --tsinghua-images "$timgs" --stanford-images "$simgs"
echo "[$(date +%H:%M:%S)] STANFORD PIPELINE DONE"
