#!/usr/bin/env bash
# Bring the results of a Kaggle session (scripts/kaggle_run.sh) into the work folder and the Drive mirror, then make
# the digest, the paper tables and the figures from every result, with the commands of scripts/auto_final.sh.
# Run on a (CPU) Colab runtime with Drive mounted and /content/work restored from the mirror:
#
#     bash scripts/ingest_kaggle.sh /content/kaggle_in
#
# /content/kaggle_in holds kaggle_results.sha256 and kaggle_results_small.tar, and kaggle_results_full.tar when it
# was brought too (the full archive is used when present: it also holds the checkpoints and prediction arrays).
# In order:
#   1. every archive present is checked against kaggle_results.sha256; a mismatch stops everything;
#   2. the archive is unpacked into a staging folder, never straight into the work folder;
#   3. every row of the mirror's run registry must be in the archive's registry, which started as a copy of it
#      (resume.tar) and is only ever appended to;
#   4. each experiment and result folder the archive holds replaces the old copy as a whole, never run by run, on
#      Drive and in the work folder; the old copies are moved to before_kaggle/<stamp>/ (nothing is deleted); the
#      Kaggle logs are added (a log they would overwrite is kept in before_kaggle/<stamp>/logs); the open-set
#      split goes to configs/openset; the archives themselves are kept on Drive in kaggle_runs/<stamp>/;
#   5. the digest, the tables and the figures, as at the end of scripts/auto_final.sh.
# LOCAL_ONLY=1 does steps 1–5 on the local work folder only (Drive is read, never written): reports from the small
# archive while the full one is still on its way.
# Nothing here trains, evaluates or chooses anything: the frozen selections and test results are the archive's.
set -euo pipefail
shopt -s nullglob
IN="$(cd "${1:?usage: ingest_kaggle.sh <folder with the Kaggle archives>}" && pwd)"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${WORK:-/content/work}"
DRIVE="${DRIVE-/content/drive/MyDrive/DogBreed-PhD-Research}"
PY="${PYTHON:-python}"
LOCAL_ONLY="${LOCAL_ONLY:-0}"   # 1: merge into the local work folder only and make the reports there; Drive is not written
IMAGES="${TSINGHUA_IMAGES:-$DRIVE/raw/tsinghua_dogs/low-resolution.zip}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
export DOGBREED_MANIFESTS="$WORK/manifests" DOGBREED_EXPERIMENTS="$WORK/experiments" DOGBREED_RESULTS="$WORK/results"
export DOGBREED_FIGURES="$WORK/figures" DOGBREED_CHECKPOINTS="$WORK/checkpoints"
LOGS="$WORK/logs"
[[ -d "$DRIVE/work" ]] || { say "the Drive mirror $DRIVE/work is not there (is Drive mounted?)"; exit 1; }

# 1. hashes
cd "$IN"
[[ -f kaggle_results.sha256 && -f kaggle_results_small.tar ]] || { say "kaggle_results.sha256 or kaggle_results_small.tar missing in $IN"; exit 1; }
if [[ -f kaggle_results_full.tar ]]; then
  sha256sum -c kaggle_results.sha256; TAR=kaggle_results_full.tar
else
  grep -E '[ *]kaggle_results_small\.tar$' kaggle_results.sha256 | sha256sum -c -; TAR=kaggle_results_small.tar
fi
say "archive: $TAR"

# 2. staging
ST="$WORK/kaggle_stage_$STAMP"
mkdir -p "$ST"
tar -xf "$IN/$TAR" -C "$ST"
[[ -f "$ST/experiments/registry.csv" ]] || { say "the archive has no experiments/registry.csv"; exit 1; }

# 3. the registry only grew
missing=$(grep -vxF -f "$ST/experiments/registry.csv" "$DRIVE/work/experiments/registry.csv" | wc -l || true)
(( missing == 0 )) || { say "$missing rows of the mirror's registry are not in the archive's registry; stopping"; exit 1; }
say "registry: $(($(wc -l < "$ST/experiments/registry.csv") - $(wc -l < "$DRIVE/work/experiments/registry.csv"))) new rows"

# 4. replace whole folders, keep the old ones
merge_files() {  # copy every file under $1 to the same place under $2; a file it would change is first kept under $3
  local src="$1" dst="$2" keep="$3" f rel
  while IFS= read -r -d '' f; do
    rel="${f#"$src"/}"
    if [[ -e "$dst/$rel" ]] && ! cmp -s "$f" "$dst/$rel"; then
      mkdir -p "$(dirname "$keep/$rel")"; cp -p "$dst/$rel" "$keep/$rel"
    fi
    mkdir -p "$(dirname "$dst/$rel")"; cp -p "$f" "$dst/$rel"
  done < <(find "$src" -type f -print0)
}
replace() {  # $1 = destination work folder (Drive mirror or local)
  local dst="$1" old="$1/before_kaggle/$STAMP" sub name
  for sub in experiments results; do
    mkdir -p "$old/$sub" "$dst/$sub"
    for d in "$ST/$sub"/*/; do
      name="$(basename "$d")"
      [[ "$sub/$name" == results/digest ]] && continue
      if [[ -e "$dst/$sub/$name" ]]; then mv "$dst/$sub/$name" "$old/$sub/$name"; fi
      cp -r "$d" "$dst/$sub/$name"
    done
  done
  cp "$dst/experiments/registry.csv" "$old/experiments/registry.csv" 2>/dev/null || true
  cp "$ST/experiments/registry.csv" "$dst/experiments/registry.csv"
  mkdir -p "$dst/logs" "$old/logs"
  merge_files "$ST/logs" "$dst/logs" "$old/logs"
  for f in openset_config openset_backup; do
    if [[ -d "$ST/$f" ]]; then mkdir -p "$dst/$f"; merge_files "$ST/$f" "$dst/$f" "$old/$f"; fi
  done
  if [[ -f "$ST/requirements-kaggle-full.txt" ]]; then cp -p "$ST/requirements-kaggle-full.txt" "$dst/"; fi
  say "replaced in $dst (old copies in $old)"
}
if (( LOCAL_ONLY )); then
  say "LOCAL_ONLY=1: the Drive mirror is read, never written"
else
  replace "$DRIVE/work"
  mkdir -p "$DRIVE/kaggle_runs/$STAMP"
  cp "$IN"/kaggle_results* "$DRIVE/kaggle_runs/$STAMP/"
  (cd "$DRIVE/kaggle_runs/$STAMP" && sha256sum -c kaggle_results.sha256 --ignore-missing)
fi
replace "$WORK"
if [[ -d "$ST/openset_config" ]]; then mkdir -p "$ROOT/configs/openset"; cp "$ST/openset_config"/*.json "$ROOT/configs/openset/"; fi
ls "$ROOT/configs/openset" 2>/dev/null || true

# 5. the reports, from every result
cd "$ROOT"
mkdir -p "$LOGS"
rc=0
"$PY" -I scripts/digest.py --out "$WORK/results/digest" \
  --curves tsinghua_main/linear_s0,tsinghua_main/rr_class_s0,tsinghua_main/rr_rival_s0,tsinghua_main/rr_rival_soft_s0,tsinghua_main/rr_pairwise_s0 \
  > "$LOGS/digest_ingest_$STAMP.log" 2>&1 || rc=1
say "digest rc=$rc"
"$PY" -I scripts/make_tables.py --digest "$WORK/results/digest" --out paper/generated > "$LOGS/tables_ingest_$STAMP.log" 2>&1 || rc=2
say "tables rc=$rc"
"$PY" -I scripts/make_figures.py --gen paper/generated --out paper/figures \
  --explain "$WORK/results/tsinghua_main/explain/shown_maps.json" --images "$IMAGES" --png \
  > "$LOGS/figures_ingest_$STAMP.log" 2>&1 || rc=3
say "figures rc=$rc"
if (( LOCAL_ONLY )); then
  say "INGEST END (local only) rc=$rc stamp=$STAMP"; exit $rc
fi
if (( rc == 0 )) && [[ -d "$WORK/results/digest" ]]; then
  if [[ -d "$DRIVE/work/results/digest" ]]; then
    mkdir -p "$DRIVE/work/before_kaggle/$STAMP/results"
    mv "$DRIVE/work/results/digest" "$DRIVE/work/before_kaggle/$STAMP/results/digest"
  fi
  cp -r "$WORK/results/digest" "$DRIVE/work/results/digest"
  say "new digest on Drive (the old one in before_kaggle/$STAMP/results/digest)"
else
  say "reports incomplete: the digest on Drive is left as it was"
fi
merge_files "$LOGS" "$DRIVE/work/logs" "$DRIVE/work/before_kaggle/$STAMP/logs"
say "INGEST END rc=$rc stamp=$STAMP"
exit $rc
