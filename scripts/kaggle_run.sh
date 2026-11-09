#!/usr/bin/env bash
# Finish the secondary experiments on a Kaggle GPU session (the Colab GPU quota ran out on 2026-11-08; research
# log, "Moving the remaining GPU work to Kaggle"). From a Kaggle notebook with Internet on, a GPU accelerator, and
# the private dataset that holds repo.tgz, resume.tar and expected_sha256.txt attached:
#
#     !mkdir -p /kaggle/working/DogBreed-PhD-Research
#     !tar -xzf $(find /kaggle/input -name repo_tgz.bin | head -1) -C /kaggle/working/DogBreed-PhD-Research
#     !bash /kaggle/working/DogBreed-PhD-Research/scripts/kaggle_run.sh
#
# The two archives are uploaded as repo_tgz.bin and resume_tar.bin (Kaggle unpacks .tgz and .tar uploads on its
# own); the original names are accepted too. Without a dataset, the files can be downloaded to a folder named by
# RESUME_IN (each checked against its SHA-256 first).
# In order:
#   1. every /content path the Colab scripts use is linked into /kaggle, so each command runs exactly as on Colab;
#   2. the repository is installed (timm pinned as on Colab; nothing else is upgraded);
#   3. resume.tar, built from the Drive mirror (manifests, saved token PCA bases, the main experiment's frozen heads
#      with their validation predictions, the finished runs of the two interrupted experiments, the queue logs), is
#      unpacked into the work folder;
#   4. Tsinghua Dogs is downloaded from the official site; each archive must match the SHA-256 of the copy the Colab
#      runs used (expected_sha256.txt), or nothing runs;
#   5. queue a (GPU 0) and queue c (GPU 1; after queue a when there is one GPU) of configs/queues/kaggle/ run beside
#      the freeze-and-evaluate watcher (scripts/auto_final.sh), as on Colab; the watcher ends with the efficiency
#      report;
#   6. what goes back to Drive is packed: kaggle_results_small.tar (no checkpoints or prediction arrays) and
#      kaggle_results_full.tar (everything), both in /kaggle/working, with a copy of the logs. This also happens
#      when the script stops early.
# The work folder, the images and the feature caches live on the scratch disk ($SCRATCH): only the packed results
# and the logs are kept as the notebook's output. A session is stopped at 12 h, so after DEADLINE_MIN minutes
# (default 630) the queues and the watcher are stopped and what exists is packed; an unfinished run keeps its
# last.pt and resumes in a later session.
# STAGE=setup stops after step 4 (a quick check of an interactive session); the default runs everything.
set -uo pipefail
STAGE="${STAGE:-all}"
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
RESUME_IN="${RESUME_IN:-/kaggle/input}"         # where resume.tar and expected_sha256.txt are (a dataset, or a download)
IN_TAR="$(find "$RESUME_IN" \( -name resume.tar -o -name resume_tar.bin \) 2>/dev/null | head -1)"
[[ -f "$IN_TAR" ]] || { echo "resume.tar (resume_tar.bin) not found under $RESUME_IN"; exit 1; }
IN="$(dirname "$IN_TAR")"
T0="$(date +%s)"
REPO=/kaggle/working/DogBreed-PhD-Research
SCRATCH="${SCRATCH:-/kaggle/tmp}"            # work folder, feature caches and images: never saved with the outputs
WORK="$SCRATCH/work"
OUT=/kaggle/working
mkdir -p "$WORK/logs" "$SCRATCH/features" "$SCRATCH/raw/tsinghua_dogs" /content/data

# 1. the Colab paths
ln -sfn "$REPO" /content/DogBreed-PhD-Research
ln -sfn "$WORK" /content/work
ln -sfn "$SCRATCH/raw" /content/data/raw
ln -sfn "$SCRATCH/features" /content/features
export WORK DRIVE="" DOGBREED_RAW=/content/data/raw DOGBREED_MANIFESTS=/content/work/manifests
export DOGBREED_FEATURES=/content/features DOGBREED_EXPERIMENTS=/content/work/experiments
export DOGBREED_RESULTS=/content/work/results DOGBREED_FIGURES=/content/work/figures
export DOGBREED_CHECKPOINTS=/content/work/checkpoints LOGS=/content/work/logs
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_MININTERVAL=30
LOG="$WORK/logs/kaggle_run.log"
exec > >(tee -a "$LOG") 2>&1
say "kaggle_run.sh stage=$STAGE; input $IN"

# 6, defined here so that it also runs when the script stops early: what goes back to Drive
pack() {
  local rc=$?
  trap - EXIT
  say "packing (exit code $rc)"
  cp -r "$REPO/configs/openset" "$WORK/openset_config" 2>/dev/null
  rm -rf "$OUT/logs"; cp -r "$WORK/logs" "$OUT/logs" 2>/dev/null
  [[ "$STAGE" == setup ]] && { say "setup stage: logs copied to $OUT/logs"; return; }
  cd "$WORK" || return
  local sel
  sel=$(ls -d experiments/tsinghua_ablate* experiments/tsinghua_openset experiments/tsinghua_tok32 \
          experiments/tsinghua_tok64 experiments/tsinghua_convnext_tiny* experiments/tsinghua_clip_vitb16 \
          experiments/registry.csv results/tsinghua_ablate results/tsinghua_openset results/tsinghua_tok32 \
          results/tsinghua_tok64 results/tsinghua_convnext_tiny results/tsinghua_clip_vitb16 results/drift \
          results/efficiency logs openset_config openset_backup requirements-kaggle-full.txt 2>/dev/null)
  tar --exclude='*.pt' --exclude='*.npz' -cf "$OUT/kaggle_results_small.tar" $sel
  tar -cf "$OUT/kaggle_results_full.tar" $sel
  (cd "$OUT" && sha256sum kaggle_results_small.tar kaggle_results_full.tar > kaggle_results.sha256)
  ls -l "$OUT"/kaggle_results_*.tar
  say "KAGGLE RUN END"
}
trap pack EXIT
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
df -h /kaggle/working "$SCRATCH" | sed 1d
free -g | sed -n 2p

# 2. the repository
cd "$REPO" || exit 1
python -m pip install -q --no-deps -e . && python -m pip install -q "timm==1.0.31" || { say "install failed"; exit 1; }
python - <<'PY'
import sys, torch, timm, numpy, sklearn, scipy, pandas
print("python", sys.version.split()[0], "| torch", torch.__version__, "cuda", torch.version.cuda,
      "cudnn", torch.backends.cudnn.version(), "| timm", timm.__version__, "| numpy", numpy.__version__,
      "| sklearn", sklearn.__version__, "| scipy", scipy.__version__, "| pandas", pandas.__version__)
PY
python -m pip freeze > "$WORK/requirements-kaggle-full.txt"

# 3. the saved state
if [[ ! -e "$WORK/.resume_unpacked" ]]; then
  tar -xf "$IN_TAR" -C "$WORK" && touch "$WORK/.resume_unpacked" || { say "resume.tar failed"; exit 1; }
fi
rsync -a "$WORK/pca_backup/" /content/features/
say "state: $(ls "$WORK/experiments" | tr '\n' ' ')"
find /content/features -name pca.npz

# 4. the images, checked against the copy the Colab runs used
bash scripts/download_data.sh tsinghua || { say "download failed"; exit 1; }
while read -r sum name; do
  got=$(sha256sum "/content/data/raw/tsinghua_dogs/$name" | cut -d' ' -f1)
  if [[ "$got" != "$sum" ]]; then say "SHA-256 MISMATCH $name: $got, expected $sum"; exit 1; fi
  say "sha256 ok $name"
done < "$IN/expected_sha256.txt"
[[ "$STAGE" == setup ]] && { say "SETUP DONE"; exit 0; }


# 5. the queues and the watcher
JOBS_LEFT="a/ablate:tsinghua_ablate a/openset:tsinghua_openset c/bb_convnext_tiny:tsinghua_convnext_tiny
           c/bb_clip_vitb16:tsinghua_clip_vitb16 a/tok32:tsinghua_tok32 a/tok64:tsinghua_tok64"
if (( NGPU >= 2 )); then
  CUDA_VISIBLE_DEVICES=0 setsid nohup bash scripts/queue.sh configs/queues/kaggle/a.txt > "$LOGS/queue_a_kaggle.log" 2>&1 < /dev/null &
  CUDA_VISIBLE_DEVICES=1 setsid nohup bash scripts/queue.sh configs/queues/kaggle/c.txt > "$LOGS/queue_c_kaggle.log" 2>&1 < /dev/null &
else
  setsid nohup bash -c "bash scripts/queue.sh configs/queues/kaggle/a.txt > $LOGS/queue_a_kaggle.log 2>&1;
                        bash scripts/queue.sh configs/queues/kaggle/c.txt > $LOGS/queue_c_kaggle.log 2>&1" < /dev/null > /dev/null 2>&1 &
fi
say "queues started on $NGPU GPU(s)"
CUDA_VISIBLE_DEVICES=0 AUTO_FINAL_JOBS="$JOBS_LEFT" AUTO_FINAL_REPORTS=efficiency \
  TSINGHUA_IMAGES=/content/data/raw/tsinghua_dogs/low-resolution.zip \
  bash scripts/auto_final.sh > "$LOGS/auto_final_kaggle.log" 2>&1 < /dev/null &
AF=$!
DEADLINE_MIN="${DEADLINE_MIN:-630}"
n=0
while kill -0 "$AF" 2>/dev/null; do
  sleep 60; n=$((n + 1))
  el=$(( ($(date +%s) - T0) / 60 ))
  if (( n % 10 == 0 )); then
    say "status at ${el} min | a: $(tail -n 1 "$LOGS/queue_a_kaggle.log" 2>/dev/null | cut -c1-70) | c: $(tail -n 1 "$LOGS/queue_c_kaggle.log" 2>/dev/null | cut -c1-70) | watcher: $(tail -n 1 "$LOGS/auto_final_kaggle.log" 2>/dev/null | cut -c1-60) | free $(df -h "$SCRATCH" | awk 'NR==2 {print $4}')"
  fi
  if (( el >= DEADLINE_MIN )); then
    say "DEADLINE: ${el} min since the start; stopping the queues and the watcher (unfinished runs keep last.pt)"
    pkill -f "scripts/auto_final.sh"; pkill -f "scripts/queue.sh"; pkill -f "run_heads.py"; pkill -f "final_eval.py"
    pkill -f "extract_"; sleep 30
    break
  fi
done
say "watcher finished: $(tail -n 1 "$LOGS/auto_final_kaggle.log")"
grep -h "SKIP\|FINAL DONE\|FINAL FAILED" "$LOGS/auto_final_kaggle.log"
exit 0
