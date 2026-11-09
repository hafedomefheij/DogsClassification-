#!/usr/bin/env bash
# Pack what a new GPU runtime needs to finish the secondary experiments without repeating a finished run (research
# log, "Moving the remaining GPU work to Kaggle"), from the Drive mirror:
#   resume.tar           manifests (the splits), the saved token PCA bases, the queue and watcher state, the run
#                        registry, the main experiment's frozen heads (best.pt, validation predictions, summaries,
#                        configuration and learning-rate selection: the rates the ablations reuse, and the drift
#                        check), and every finished run of the two interrupted experiments (tsinghua_ablate,
#                        tsinghua_convnext_tiny); nothing from any test split
#   expected_sha256.txt  SHA-256 of the three Tsinghua archives the Colab runs used
#
#   bash scripts/make_resume_bundle.sh /content/drive/MyDrive/DogBreed-PhD-Research /content/resume_out
set -euo pipefail
SRC="$1/work"; RAWD="$1/raw/tsinghua_dogs"; OUT="$2"
mkdir -p "$OUT"
list="$OUT/resume_files.txt"
cd "$SRC"
{
  find manifests pca_backup -type f
  find logs -type f \( -path 'logs/queue_*' -o -path 'logs/auto_final/*' \)
  echo experiments/registry.csv
  find experiments/tsinghua_main -maxdepth 1 -type f \( -name 'config_*.json' -o -name lr_selection.json \)
  find experiments/tsinghua_main -mindepth 2 -maxdepth 2 -type f \
       \( -name best.pt -o -name val_preds.npz -o -name summary.json \) ! -path '*_lr*'
  find experiments/tsinghua_ablate experiments/tsinghua_convnext_tiny -type f ! -name last.pt ! -name 'test_*'
} | sort > "$list"
tar -cf "$OUT/resume.tar" -T "$list"
(cd "$RAWD" && sha256sum TrainValSplit.zip low-annotations.zip low-resolution.zip) > "$OUT/expected_sha256.txt"
echo "files: $(wc -l < "$list")"
cat "$OUT/expected_sha256.txt"
sha256sum "$OUT/resume.tar"
ls -l "$OUT"
