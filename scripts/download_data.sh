#!/usr/bin/env bash
# Download the benchmark datasets into data/raw/<name>/.
#
#   bash scripts/download_data.sh stanford     # Stanford Dogs (~0.8 GB)
#   bash scripts/download_data.sh tsinghua     # Tsinghua Dogs, low-resolution release (~2.5 GB)
#   bash scripts/download_data.sh all
#
# Idempotent: a finished file gets a "<file>.ok" marker and is skipped next time;
# an interrupted file is resumed with curl -C -.  Every finished file also gets a
# "<file>.sha256" record, so the exact bytes used can be reported.
#
# Sources (checked 2026-11-08):
#   Stanford Dogs  http://vision.stanford.edu/aditya86/ImageNetDogs/  (official)
#                  On 2026-11-08 the official host (and image-net.org) refused connections
#                  from two networks, so each file falls back to the Internet Archive's
#                  capture of that same official URL ("id_" = the original bytes).  The
#                  fallback is accepted only if its SHA-256 equals the value TensorFlow
#                  Datasets recorded from the official host
#                  (tensorflow_datasets/datasets/stanford_dogs/checksums.tsv), i.e. only if
#                  it is byte-identical to the official release.
#   Tsinghua Dogs  https://cg.cs.tsinghua.edu.cn/ThuDogs/  (official; no published checksums)
# Neither dataset may be redistributed: this repository ships manifests (paths, labels,
# hashes), never the images. Stanford Dogs images come from ImageNet (non-commercial
# research use only).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="${DOGS_RAW:-${DOGBREED_RAW:-$ROOT/data/raw}}"
what="${1:-all}"

fetch_from() {  # fetch_from <url> <out> <attempts>: download, resuming after every drop
  local url="$1" out="$2" n="$3" attempt rc
  for attempt in $(seq 1 "$n"); do
    curl -fL --retry 5 --retry-delay 10 --connect-timeout 30 -sS -C - -o "$out" "$url"
    rc=$?
    [[ $rc -eq 0 ]] && return 0
    # 33 = the server refused a range request: the partial file cannot be resumed.
    [[ $rc -eq 33 ]] && rm -f "$out"
    echo "[$(date +%H:%M:%S)] attempt $attempt/$n (curl $rc) stopped at $(stat -c %s "$out" 2>/dev/null || echo 0) bytes"
    [[ $attempt -lt $n ]] && sleep 10
  done
  return 1
}

fetch() {  # fetch <out> <sha256 or -> <url> [<fallback url> ...]
  local out="$1" sum="$2" url n last
  shift 2
  if [[ -f "$out.ok" ]]; then
    # Finished earlier (or copied from Drive): record its hash once, and re-check it
    # against the official value when there is one.
    [[ -f "$out.sha256" ]] || printf '%s  %s\n%s\n' "$(sha256sum "$out" | cut -d' ' -f1)" \
      "$(basename "$out")" "source: earlier run" > "$out.sha256"
    if [[ "$sum" == - || "$(head -c 64 "$out.sha256")" == "$sum" ]]; then
      echo "skip (done): $out"; return 0
    fi
    echo "[$(date +%H:%M:%S)] $out does not match the official SHA-256; downloading again"
    rm -f "$out" "$out.ok" "$out.sha256"
  fi
  last="${*: -1}"
  for url in "$@"; do
    # A source with a fallback after it gets a few attempts; the last source gets them all.
    n="${DOWNLOAD_ATTEMPTS:-60}"; [[ "$url" != "$last" ]] && n="${PRIMARY_ATTEMPTS:-3}"
    # A source that has a fallback is probed first, so a host that is down costs seconds.
    if [[ "$url" != "$last" ]] && ! curl -sfIL -o /dev/null --connect-timeout 15 -m 40 "$url"; then
      echo "[$(date +%H:%M:%S)] unreachable, trying the next source: $url"; continue
    fi
    echo "[$(date +%H:%M:%S)] GET $url -> $out"
    fetch_from "$url" "$out" "$n" || { echo "[$(date +%H:%M:%S)] source failed: $url"; continue; }
    local got
    got="$(sha256sum "$out" | cut -d' ' -f1)"
    if [[ "$sum" != - && "$got" != "$sum" ]]; then
      echo "[$(date +%H:%M:%S)] SHA-256 MISMATCH from $url (got $got, want $sum); discarded"
      rm -f "$out"; continue
    fi
    printf '%s  %s\n%s\n' "$got" "$(basename "$out")" "source: $url" > "$out.sha256"
    touch "$out.ok"
    if [[ "$sum" == - ]]; then
      echo "[$(date +%H:%M:%S)] OK $out $(stat -c %s "$out") bytes sha256 $got"
    else
      echo "[$(date +%H:%M:%S)] OK $out $(stat -c %s "$out") bytes, SHA-256 matches the official release"
    fi
    return 0
  done
  echo "FAILED $out"
  return 1
}

status=0
if [[ "$what" == all || "$what" == stanford ]]; then
  d="$RAW/stanford_dogs"; mkdir -p "$d"
  base=http://vision.stanford.edu/aditya86/ImageNetDogs
  ia=https://web.archive.org/web
  fetch "$d/lists.tar" 34b47cacd9a98b5d150e084f24d29391c084c55272295ec65c85651bc35f4d6c \
    "$base/lists.tar" "$ia/20241221112710id_/$base/lists.tar" || status=1
  fetch "$d/annotation.tar" 662ed36f707be6b195c2fda392d8e153e83af28cedc8e2134a649f198def30fd \
    "$base/annotation.tar" "$ia/20250724000413id_/$base/annotation.tar" || status=1
  fetch "$d/images.tar" a443fc2f9a851aa1a7b8e4fa4af94b8459fce2c9f5a2f4dd4669cf9673a4ec58 \
    "$base/images.tar" "$ia/20250724000413id_/$base/images.tar" || status=1
fi
if [[ "$what" == all || "$what" == tsinghua ]]; then
  d="$RAW/tsinghua_dogs"; mkdir -p "$d"
  fetch "$d/TrainValSplit.zip"   - "https://cg.cs.tsinghua.edu.cn/ThuDogs/TrainValSplit.zip"    || status=1
  fetch "$d/low-annotations.zip" - "https://cg.cs.tsinghua.edu.cn/ThuDogs/low-annotations.zip"  || status=1
  fetch "$d/low-resolution.zip"  - "https://cloud.tsinghua.edu.cn/f/80013ef29c5f42728fc8/?dl=1" || status=1
fi
if [[ $status -eq 0 ]]; then echo "DOWNLOADS DONE ($what)"; else echo "DOWNLOADS FAILED ($what)"; fi
exit $status
