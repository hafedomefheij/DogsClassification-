#!/usr/bin/env bash
# Colab: put both datasets on the local disk, with Google Drive as a download cache, then
# build the manifests and audits.
#
#   bash scripts/colab_data.sh [tsinghua|stanford|all] [--no-prepare]
#
# 1. an archive already saved on Drive is copied to the local disk (fast, no re-download);
# 2. anything missing is downloaded from the official host (scripts/download_data.sh);
# 3. new archives are saved to Drive for the next session;
# 4. archives are unpacked where the manifest builders expect them (Tsinghua images stay
#    inside low-resolution.zip and are read from it directly);
# 5. scripts/prepare_data.py writes the manifest and the audit report.
# Environment (scripts/colab_env.py): DRIVE, DOGBREED_RAW, DOGBREED_MANIFESTS, DOGBREED_RESULTS.
# DRIVE="" (Drive not mounted) skips steps 1 and 3.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRIVE="${DRIVE-/content/drive/MyDrive/DogBreed-PhD-Research}"
use_drive=0
[[ -n "$DRIVE" && -d "$(dirname "$DRIVE")" ]] && use_drive=1
RAW="${DOGBREED_RAW:-/content/data/raw}"
export DOGS_RAW="$RAW"
what="${1:-all}"
prepare=1
[[ "${2:-}" == "--no-prepare" ]] && prepare=0
PY="${PYTHON:-python}"

files_of() {
  case "$1" in
    tsinghua) echo "TrainValSplit.zip low-annotations.zip low-resolution.zip" ;;
    stanford) echo "lists.tar annotation.tar images.tar" ;;
  esac
}

for ds in tsinghua stanford; do
  [[ "$what" == all || "$what" == "$ds" ]] || continue
  loc="$RAW/${ds}_dogs"; drv="$DRIVE/raw/${ds}_dogs"
  mkdir -p "$loc"
  [[ $use_drive -eq 1 ]] && mkdir -p "$drv"
  for f in $(files_of "$ds"); do
    if [[ $use_drive -eq 1 && -f "$drv/$f.ok" && ! -f "$loc/$f.ok" ]]; then
      echo "[$(date +%H:%M:%S)] copy from Drive: $ds/$f"
      cp "$drv/$f" "$loc/$f" && touch "$loc/$f.ok"
      [[ -f "$drv/$f.sha256" ]] && cp "$drv/$f.sha256" "$loc/$f.sha256"
    fi
  done
  bash "$ROOT/scripts/download_data.sh" "$ds" || { echo "DATA FAILED: download $ds"; exit 1; }
  for f in $(files_of "$ds"); do
    if [[ $use_drive -eq 1 && ! -f "$drv/$f.ok" ]]; then
      echo "[$(date +%H:%M:%S)] save to Drive: $ds/$f"
      cp "$loc/$f" "$drv/$f" && cp "$loc/$f.sha256" "$drv/$f.sha256" && touch "$drv/$f.ok"
    fi
  done
  echo "[$(date +%H:%M:%S)] unpack $ds"
  if [[ "$ds" == tsinghua ]]; then
    [[ -d "$loc/split/TrainAndValList" ]] || unzip -o -q "$loc/TrainValSplit.zip" -d "$loc/split"
    imgs="$loc/low-resolution.zip"
  else
    [[ -f "$loc/lists/train_list.mat" ]] || { mkdir -p "$loc/lists"; tar -xf "$loc/lists.tar" -C "$loc/lists"; }
    [[ -d "$loc/Annotation" ]] || tar -xf "$loc/annotation.tar" -C "$loc"
    [[ -d "$loc/Images" ]] || tar -xf "$loc/images.tar" -C "$loc"
    imgs="$loc/Images"
  fi
  if [[ $prepare -eq 1 ]]; then
    echo "[$(date +%H:%M:%S)] prepare $ds"
    "$PY" -I "$ROOT/scripts/prepare_data.py" --dataset "$ds" --raw "$loc" --images "$imgs" \
      --workers "${WORKERS:-4}" || { echo "DATA FAILED: prepare $ds"; exit 1; }
  fi
  echo "[$(date +%H:%M:%S)] DATA READY $ds"
done
echo "DATA DONE ($what)"
