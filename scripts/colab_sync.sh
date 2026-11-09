#!/usr/bin/env bash
# Mirror the small, valuable outputs between the Colab local disk and Google Drive.
#
#   bash scripts/colab_sync.sh push   # /content/work -> Drive (after every stage)
#   bash scripts/colab_sync.sh pull   # Drive -> /content/work (start of a new session)
#
# Feature caches are not mirrored: they are recomputed in minutes from the images.
set -u
DRIVE="${DRIVE-/content/drive/MyDrive/DogBreed-PhD-Research}"
WORK="${WORK:-/content/work}"
if [[ -z "$DRIVE" || ! -d "$(dirname "$DRIVE")" ]]; then
  echo "SYNC SKIPPED: Google Drive is not mounted"; exit 0
fi
mkdir -p "$DRIVE/work" "$WORK"
case "${1:-push}" in
  push) rsync -a --exclude 'last.pt' "$WORK/" "$DRIVE/work/" && echo "SYNC PUSH OK $(du -sh "$DRIVE/work" | cut -f1)" ;;
  pull) rsync -a "$DRIVE/work/" "$WORK/" && echo "SYNC PULL OK $(du -sh "$WORK" | cut -f1)" ;;
  *) echo "usage: colab_sync.sh push|pull"; exit 2 ;;
esac
