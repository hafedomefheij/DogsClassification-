# Rival-conditioned evidence attention for fine-grained dog-breed classification

Research code, protocol and paper for **RCEA**, a re-ranking head for fine-grained recognition. A frozen backbone
gives a first answer. RCEA then re-scores only that answer's top-K breeds, and it lets each candidate look at the
image with a query built from *what separates it from the other candidates*: its own embedding minus its rivals'
(pairwise version: one query per rival, combined with a soft minimum). The paper asks a narrow question: does
conditioning the evidence query on the rivals help, compared with queries of exactly the same size that ignore them?

* Primary dataset: **Tsinghua Dogs** (130 breeds). Secondary: **Stanford Dogs** (120 breeds).
* Backbone: frozen DINOv2 ViT-B/14 with registers (timm `vit_base_patch14_reg4_dinov2.lvd142m`), 224 px; heads
  trained on cached features. Further backbones: ResNet-50, ConvNeXt-T, CLIP ViT-B/16.
* Everything runs on one free-tier Colab T4. No end-to-end fine-tuning is run (see *Results and status*).

The design document is [`docs/design.md`](docs/design.md): candidates considered, notation and equations, the
four facts with proofs, pseudocode, complexity, and the hypotheses with their refutation rules, all written
before any training. The literature review and novelty search are in [`research_log/lit_core.md`](research_log/lit_core.md)
and [`research_log/lit_related.md`](research_log/lit_related.md). Every decision and event is logged with its
time in [`research_log/research_log.md`](research_log/research_log.md).

## Repository layout

```
src/dogbreed/            the package
  datasets/              manifests, splits, audits (hashes), image sources, corruptions, transforms
  models/                backbones (timm), heads (linear, attentive probe, class attention, re-ranker)
  modules/               rival_attention.py: the RCEA mechanism
  training/              feature extraction into a resumable cache; head training on cached features
  evaluation/            metrics (top-k, macro P/R/F1, ECE, NLL, OOD), statistics (bootstrap, McNemar),
                         efficiency; runs.py: finished runs and the freeze guard (require_freeze)
scripts/                 one script per step (see "How to reproduce"); *.sh are the drivers
configs/experiments/     one YAML per experiment (dataset, backbone, heads, seeds, lr grid or lr_selection_from)
configs/queues/          a.txt, b.txt, c.txt: the job lists scripts/queue.sh runs unattended
tests/                   unit tests (pytest): the mechanism, the freeze guard, the provenance subsets,
                         planted-copy tests of the overlap audit, the seeded evaluation of the random control
notebooks/               the three Colab notebooks, written by notebooks/make_notebooks.py
docs/design.md           the method and the pre-registered hypotheses
research_log/            literature dossiers and the chronological research log
results/                 literature_table.csv; the digest copied back from the runtime (results/digest/)
paper/                   LaTeX source (main.tex) and bibliography (refs.bib, built by merge_bib.py);
                         generated/ and figures/ are written by make_tables.py and make_figures.py
dist/                    how code was moved to the Colab VM (typed, hash-checked bundles)
```

What the pipeline writes at run time is listed under [Outputs](#outputs). `figures/` and `checkpoints/` are
defaults in `src/dogbreed/paths.py` that no current script writes to.

## Data

The datasets are **not** redistributed here; download them from their owners.

| Dataset | Source | Terms (as recorded in `lit_core.md` §2) |
|---|---|---|
| Tsinghua Dogs | official URLs on cg.cs.tsinghua.edu.cn / cloud.tsinghua.edu.cn | "available for academic purposes"; citation requested; no licence file |
| Stanford Dogs | vision.stanford.edu (official). Our copy came from Internet Archive snapshots **of the official URLs**, accepted only because the SHA-256 of each file equals the checksum TensorFlow Datasets records for the official release | inherits the ImageNet terms: non-commercial research and education only; no redistribution |

`scripts/download_data.sh` fetches both, resumes interrupted files, and records each file's SHA-256 and source;
a Stanford Dogs file from any source is kept only if it matches the official checksum (Tsinghua Dogs publishes
none). `scripts/prepare_data.py` writes the manifests (one row per image with label, official split, our split,
decode status and hashes) and the within-dataset audit described [below](#the-leakage-and-provenance-audit).

The validation split is a stratified 10 % of the official training images (seed 0). All model selection uses
it. The official test split (for Tsinghua Dogs, the official validation list) is read only by the locked
evaluation, after the selection freeze.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e . --no-deps
python -m pytest -q tests
```

`requirements.txt` gives lower bounds. [`requirements-colab.txt`](requirements-colab.txt) pins the exact versions of
the runtime that produced every result (Python 3.13.16, torch 2.14.1+cu130, timm 1.0.31, a Tesla T4), and the full
`pip freeze` of that VM is kept with the results on Drive.

The scripts run as `python -I` (isolated mode), so the package must be installed as above. Paths default to
folders inside the repository. Each one can be moved with an environment variable (`DOGBREED_RAW`,
`DOGBREED_MANIFESTS`, `DOGBREED_FEATURES`, `DOGBREED_EXPERIMENTS`, `DOGBREED_RESULTS`; also `DOGBREED_ROOT`,
`DOGBREED_DATA`, `DOGBREED_FIGURES`, `DOGBREED_CHECKPOINTS`; see `src/dogbreed/paths.py`). On Colab,
`scripts/colab_env.py`, exec'd in the notebook kernel, sets them: the images (`/content/data/raw`) and the
feature caches (`/content/features`) stay on the local disk, everything else goes under `/content/work`, which
`scripts/colab_sync.sh push|pull` mirrors to Google Drive when it is mounted (`last.pt` files excluded).

## How to reproduce

The study runs on one free-tier Colab T4, driven by three notebooks:
[`01_data_and_features`](notebooks/01_data_and_features.ipynb) (steps 1–2),
[`02_train_and_select`](notebooks/02_train_and_select.ipynb) (steps 3–4) and
[`03_evaluate_and_report`](notebooks/03_evaluate_and_report.ipynb) (steps 5–10). Each starts with the same setup
cell: it puts the repository at `/content/DogBreed-PhD-Research` (the queue files `cd` there), installs the
package, runs the tests, sets the paths with `scripts/colab_env.py` and restores earlier sessions' work from
Drive. The commands below are what the notebooks run, from the repository root, in that environment. Outside
Colab, set the `DOGBREED_*` paths (`colab_data.sh` falls back to `/content/data/raw`), pass the image paths to
the drivers (their defaults are `/content/...` paths), set `LOGS` for `queue.sh` and `TSINGHUA_IMAGES` for
`final_all.sh`, and note that the queue files hard-code `cd /content/DogBreed-PhD-Research`.

**Long jobs on Colab.** A notebook cell `!nohup cmd &` keeps the cell, and with it the kernel, busy until the job
ends, because the job holds the cell's terminal. The notebooks therefore start every long job detached, with all
three streams redirected, for example

```
!setsid nohup bash -c 'bash scripts/run_experiment.sh configs/experiments/tsinghua_main.yaml' > /content/work/logs/tsinghua_main.log 2>&1 < /dev/null &
```

The cell returns at once and a watch cell tails the log. The same commands can be typed into Colab's Terminal
panel instead; a terminal does not inherit the kernel's environment, so export the variables
`scripts/colab_env.py` sets first, or the scripts will write into the repository's own folders.

1. **Data, manifests, within-dataset audit.**
   ```bash
   bash scripts/colab_data.sh all                 # [tsinghua|stanford|all] [--no-prepare]
   ```
   Copies the archives already saved on Drive, downloads the rest (`download_data.sh`), saves new archives to
   Drive, unpacks them and runs `scripts/prepare_data.py` for each dataset. `--no-prepare` stops after unpacking
   (after a recycled runtime, when the manifests come back from Drive).

2. **Feature caches and the cross-dataset audit.**
   ```bash
   bash scripts/extract_all.sh tsinghua dinov2_vitb14_reg /content/data/raw/tsinghua_dogs/low-resolution.zip 256 224
   bash scripts/pipeline_stanford.sh              # [stanford images] [tsinghua images] [backbone]
   ```
   `extract_all.sh <dataset> <backbone> <images> [token_pca=256|none] [img_size=224]` caches the orig view of
   train, val and test (the token PCA is fitted here, on 4,000 training images only), hflip of train, then the
   five automatic and the five annotation-assisted test views (`AUTO_VIEWS` / `ANN_VIEWS` override the lists, an
   empty list skips the step). Every cache resumes after an interruption. `pipeline_stanford.sh` builds the
   Stanford caches in Stanford's own PCA basis (its annotation-assisted views are the body-box ones: Stanford Dogs
   has no head boxes), the Stanford test split in the Tsinghua PCA basis for the transfer study, and then runs
   `scripts/cross_overlap.py` ([below](#the-leakage-and-provenance-audit)). If training jobs fill the GPU, run the
   audit alone with `--device cpu`. The other backbones' Tsinghua caches (orig and hflip only) come from queue B.

3. **Training: `tsinghua_main`, then queues A, B and C, side by side.**
   ```bash
   bash scripts/run_experiment.sh configs/experiments/tsinghua_main.yaml   # [--only tag,tag]
   bash scripts/queue.sh configs/queues/a.txt
   bash scripts/queue.sh configs/queues/b.txt
   bash scripts/queue.sh configs/queues/c.txt     # waits for queue B's extractions
   ```
   `run_experiment.sh` runs `run_heads.py --select-lr` (the grid on seed 0; validation top-1 picks the rate, ties
   go to the lower validation NLL; `lr_selection.json`), then `run_heads.py` for every seed at the chosen rates.
   Finished runs are skipped; an interrupted run resumes from its `last.pt`. `queue.sh` runs a queue file's
   `name | command` lines in order, each once, leaving `<name>.done` or `<name>.failed` in `$LOGS/queue_<a|b>/`
   (`LOGS` defaults to `/content/work/logs`) and the job's output in `$LOGS/<name>.log`. It picks up lines added
   while it runs and exits once every job is marked and the file has an `END` line. To retry a failed job, delete
   its `.failed` marker (and start the queue again if it has exited).

   | Queue | Jobs, in order |
   |---|---|
   | A (`a.txt`): Tsinghua-sized jobs, about 9 GB of tokens on the GPU each, one at a time | `wait_main` (until `tsinghua_main` has finished) → `ablate` (`tsinghua_ablate`) → `openset_split` (`make_openset_split.py --dataset tsinghua --unknown 30 --seed 0`) → `openset` (`tsinghua_openset`) → `wait_bb` (until queue B's `extract_clip_vitb16` is marked) → `tok32`, `tok64` |
   | B (`b.txt`): Stanford-sized jobs, alongside A | `wait_stanford` (until `pipeline_stanford.sh` has finished) → `stanford_main` → `extract_resnet50`, `extract_convnext_tiny`, `extract_clip_vitb16` |
   | C (`c.txt`): the other backbones, alongside A | `wait_b` (until queue B's `extract_clip_vitb16` is marked) → `bb_resnet50`, `bb_convnext_tiny`, `bb_clip_vitb16`, each with its own learning-rate grid (moved unchanged from queue A during the recorded run, to use the GPU share B had left idle; `wait_b` was added afterwards for a fresh start) |

   Start `tsinghua_main` first and exactly as written: `wait_main` recognises it by its command line
   (`pgrep -f "run_experiment.sh configs/experiments/tsinghua_main.yaml"`). If it is not running, queue A goes
   straight on to the ablations, which then fail for want of its learning rates. The ablation, open-set and
   token-width configs borrow those rates (`lr_selection_from: tsinghua_main`, `lr_like` for renamed variants)
   and select nothing themselves; each other backbone has its own grid.

4. **Freeze model selection**, once per experiment, when every configured run has finished (the experiment's
   log ends with `TRAINING DONE`).
   ```bash
   python -I scripts/freeze.py --experiment tsinghua_main          # [--expect-seeds 3] [--force]
   ```
   Writes `experiments/<exp>/freeze.json`: the time (UTC), each head's learning rate, and for every seed run its
   validation metrics, best epoch and the SHA-256 of its `best.pt`. The printed summary goes into the section
   "Model selection freeze" of `research_log/research_log.md` before any test evaluation of that experiment.
   `freeze.py` refuses a head with fewer than `--expect-seeds` finished seeds and a second freeze without
   `--force`. It only sees heads with at least one finished seed, so do not freeze an experiment that is still
   training; notebook 2 checks every run of the config first.

5. **Locked test evaluation**, after the freeze is recorded.
   ```bash
   bash scripts/final_all.sh tsinghua_main > /content/work/logs/final_tsinghua_main.log 2>&1
   ```
   One experiment per call; every step refuses to run without a matching freeze, and the log ends with
   `FINAL DONE <exp>`. `PYTHON` and `TSINGHUA_IMAGES` (default `/content/data/raw/tsinghua_dogs/low-resolution.zip`)
   can be set. The views, the comparison pairs and their order are fixed in the script (`docs/design.md` §8–9):

   | Experiment | What `final_all.sh` runs, in order |
   |---|---|
   | `tsinghua_main` | `final_eval.py` on orig with the 22 main pairs (each proposed head against the three controls, all-class attention and B0; pairwise and rival-soft against rival; the control chain random → centred → class; class query against B0; rival and pairwise against the attentive probe); on the automatic corruptions `blur,noise,jpeg,lowres,occlude`; separately, on the annotation-assisted views `head_occlude,bg_gray,bg_blur,crop_body,crop_head`, both with 7 pairs against B0 and the class query; `subset_eval.py --view orig` with the 22 main pairs; `transfer_eval.py`; `explain.py --images $TSINGHUA_IMAGES` |
   | `stanford_main` | the same three `final_eval.py` passes, with `bg_gray,bg_blur,crop_body` as the annotation-assisted views |
   | `tsinghua_ablate` | `final_eval.py` on orig: rival against class query at K = 2, 3, 10; pairwise against rival at K = 3, 10 |
   | `tsinghua_openset` | `ood_eval.py`: known against held-out breeds (AUROC, FPR at 95 % TPR, closed-set top-1) |
   | `tsinghua_resnet50`, `tsinghua_convnext_tiny`, `tsinghua_clip_vitb16` | `final_eval.py` on orig: rival and pairwise against class query, pairwise against rival, rival and class query against B0 |
   | `tsinghua_tok32`, `tsinghua_tok64` | `final_eval.py` on orig: rival against class query, rival and class query against the attentive probe |

   **Unattended.** `scripts/auto_final.sh`, started once beside the queues, does steps 4 and 5 for every
   secondary experiment (ablation, open set, the three backbones, token width) as soon as queue A or C marks its
   training job done, one experiment at a time; when all are finished it runs steps 6–9. It chooses nothing: the
   plans are the ones in the table above. The freeze printouts are kept in `$LOGS/freeze_<exp>.log` and copied into
   the research log afterwards (the one deviation from step 4, declared in the log before any of these freezes).
   ```bash
   setsid nohup bash scripts/auto_final.sh > /content/work/logs/auto_final.log 2>&1 < /dev/null &
   ```

   `final_eval.py` fits each run's temperature on its validation predictions, reports the mean, std and 95 % CI
   over seeds, compares pairs seed by seed on the same test images, defines the hard pairs on the linear head's
   validation predictions, and saves each run's test predictions (`test_<view>.npz`). `subset_eval.py` re-scores
   those saved predictions on the provenance subsets and trains nothing; `transfer_eval.py` labels the
   de-duplicated Stanford test images with the Tsinghua-trained heads; `explain.py` records where the attention
   of each head falls.

6. **Efficiency**, with nothing else on the GPU; it reads no test data.
   ```bash
   python -I scripts/efficiency_report.py --config configs/experiments/tsinghua_main.yaml   # [--iters 50] [--device ...]
   ```
   Parameters, FLOPs (`torch.utils.flop_counter`; a multiply-add = 2 FLOPs), latency (median of `--iters` timed
   passes after 10 warm-up passes, on the runtime's device, whose name is recorded) and peak GPU memory: every
   head on random inputs of the cached shapes (batch 1 and 256), the frozen backbone (batch 1 and 64, fp16
   autocast on a T4), and the full single-image pipeline (backbone → token PCA → head). The paper's table reads
   `tsinghua_main`.

7. **Digest.**
   ```bash
   python -I scripts/digest.py --curves tsinghua_main/linear_s0,tsinghua_main/rr_class_s0,tsinghua_main/rr_rival_s0,tsinghua_main/rr_rival_soft_s0,tsinghua_main/rr_pairwise_s0
   ```
   Collects every result into small CSV/JSON files in `results/digest/` (training runs, learning rates, test
   summaries and per-seed rows, comparisons, hard pairs, per-class accuracy, the provenance subsets and their
   comparisons, open set, transfer, efficiency, attention statistics, curves, audit numbers) and writes
   `MANIFEST.txt` with the SHA-256 and size of each. `--curves` takes `<experiment>/<run>` folders; the curves
   figure draws seed 0 of B0, the class query and the proposed head picked on validation, so the list holds all
   three proposed heads, Tsinghua runs only. The digest is what leaves the runtime: copy it into the
   repository's `results/digest/` and check every file against the manifest, for example
   `cd results/digest && awk '{print $1 "  " $3}' MANIFEST.txt | sha256sum -c -`.

8. **Tables.**
   ```bash
   python -I scripts/make_tables.py --digest results/digest --out paper/generated   # on Colab: --digest $DOGBREED_RESULTS/digest
   ```
   Reads the digest and nothing else, so the same command gives the same tables on the runtime and on any copy of
   the digest. Writes `tab_*.tex` (one booktabs tabular per table, input by `paper/main.tex`), `numbers.json` (the
   numbers the text quotes, H1–H7 judged by the rule of `docs/design.md` §8, and the tables skipped for missing
   inputs) and `figdata.json`.

9. **Figures.**
   ```bash
   python -I scripts/make_figures.py --gen paper/generated --out paper/figures \
       --explain results/tsinghua_main/explain/shown_maps.json --images data/raw/tsinghua_dogs/low-resolution.zip
   ```
   Vector PDFs `fig_main`, `fig_compare`, `fig_robust`, `fig_ksweep`, `fig_perclass`, `fig_curves` and
   `fig_explain` from `figdata.json` (`--png` adds previews); the attention figure also needs the maps `explain.py`
   saved and the original images. A figure whose data is missing is skipped and listed. On Colab the two paths are
   `$DOGBREED_RESULTS/tsinghua_main/explain/shown_maps.json` and `/content/data/raw/tsinghua_dogs/low-resolution.zip`.

10. **Paper**, on the VM with TeX Live from apt.
    ```bash
    apt-get install -y latexmk texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended
    cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
    ```
    `paper/refs.bib` is built from the two verified bibliographies by `python -I paper/merge_bib.py`. `paper/`
    is in the repository checkout, which `colab_sync.sh` does not mirror, so notebook 3 copies `generated/`,
    `figures/` and `main.pdf` into `/content/work/paper` before it pushes to Drive.

**After a recycled runtime.** The setup cell brings `/content/work` back from Drive; the feature caches are lost.
The last section of notebook 1 rebuilds them without downloading again (`colab_data.sh all --no-prepare`) and
without re-running the overlap audit, whose `results/audit/cross_pairs.csv` defines the provenance subsets. The
notebooks keep a copy of every token PCA (`pca.npz`) under `/content/work/pca_backup/` and put it back first:
`extract_features.py` reuses an existing `pca.npz`, so the rebuilt caches are in the basis the heads were trained
in. `configs/openset/` is in the repository checkout and is written again with the same command (same seed, same
breeds). Then start the training cells again: finished runs are skipped and jobs marked done are not repeated; a
run cut off mid-way restarts from its first epoch, since `last.pt` is not mirrored.

## Outputs

Locations relative to the repository, with the Colab location set by `scripts/colab_env.py` in brackets; on
Colab every `results/` folder is under `/content/work/results`.

| Folder | Written by | Holds |
|---|---|---|
| `data/raw/` (`/content/data/raw`) | `download_data.sh`, `colab_data.sh` | the archives, each with `.sha256` (hash and source) and `.ok` markers, and the unpacked files; the Tsinghua images are read from `low-resolution.zip` directly |
| `data/manifests/` (`/content/work/manifests`) | `prepare_data.py` | `tsinghua.csv`, `stanford.csv`: one row per image, keyed by the `id` every cache row and prediction file uses |
| `features/` (`/content/features`, not mirrored) | `extract_features.py` | `<dataset>/<backbone>_224_pca256/<view>/` caches (`meta.json`, `ids.npy`, `global.npy`, `tokens.npy`, `done.npy`) with the train-fitted token PCA (`pca.npz`, `pca.json`) one level up; `stanford/<backbone>_224_pca256_tsinghua/orig/` is the Stanford test split in the Tsinghua basis |
| `experiments/` (`/content/work/experiments`) | `run_heads.py`, `freeze.py`, `final_eval.py` | `registry.csv` (one row per run); per experiment `config_<time>.json`, `lr_selection.json`, the grid runs `<tag>_lr<rate>_s0/`, the seed runs `<tag>_s<seed>/` (`curves.csv`, `best.pt`, `summary.json`, `val_preds.npz`; after the locked evaluation `test_<view>.npz` and `test_metrics_<view>.json`) and `freeze.json` |
| `results/audit/` | `prepare_data.py`, `cross_overlap.py`, `ilsvrc_overlap.py` | `tsinghua.json`, `stanford.json` (within-dataset audits); `cross_tsinghua_stanford.json` (counts, breed correspondence, threshold sensitivity), `cross_pairs.csv` (every candidate pair with both signals), `stanford_clean_test.json` (the transfer test list), `cross_nn_hist.png`, `cross_montage_cos*.png`; `ilsvrc_overlap.json` (Stanford Dogs against the ILSVRC-2012 training list) |
| `results/<exp>/<view>/` | `final_eval.py`, `subset_eval.py` | `runs.csv`, `summary.csv`, `summary.json`, `comparisons.json`, `hard_pairs.json`, confusion matrices (`.npy`); `subsets.json` in `tsinghua_main/orig/` |
| `results/<exp>/` | `ood_eval.py`, `transfer_eval.py`, `explain.py` | `ood_orig.json`, `ood_orig_runs.csv` (`tsinghua_openset`); `transfer_stanford.json`, `transfer_stanford_runs.csv` and `explain/` with `box_stats.json`, `shown_maps.json`, `explain_test.png` (`tsinghua_main`) |
| `results/efficiency/` | `efficiency_report.py` | `<exp>.json`, `<exp>.csv` |
| `results/digest/` | `digest.py` | the CSV/JSON digest and `MANIFEST.txt` |
| `paper/generated/`, `paper/figures/` (in the checkout; notebook 3 copies them to `/content/work/paper`) | `make_tables.py`, `make_figures.py` | `tab_*.tex`, `numbers.json`, `figdata.json`; `fig_*.pdf` (and `.png` with `--png`) |
| `configs/openset/` (in the checkout, not mirrored) | `make_openset_split.py` (queue A) | `tsinghua_unknown30_s0.json`: the 100 known and 30 held-out breeds |
| `/content/work/logs/` (Colab) | the detached jobs, `queue.sh`, `auto_final.sh` | one log per job; `queue_a/`, `queue_b/`, `queue_c/` with the `.done` / `.failed` markers; `freeze_<exp>.log`, `final_<exp>.log` and `auto_final/` (its `.done` / `.skipped` markers) |

## The leakage and provenance audit

| Overlap | Handling |
|---|---|
| Near-duplicates inside a dataset (`prepare_data.py`: SHA-1, pHash, dHash) | a training image that duplicates a test image exactly or nearly (pHash Hamming ≤ 4) is excluded from training: 86 Tsinghua and 73 Stanford images. Near pairs with different labels (113 Tsinghua, 48 Stanford) are recorded in the audit report, not relabelled. |
| Stanford Dogs is a subset of the ImageNet-1k training set; DINOv2's LVD-142M contains ImageNet-22k as is (its Table 15), and with it, by inference, the Stanford Dogs images, without labels; ResNet-50 and ConvNeXt-T are supervised on ImageNet-1k; CLIP's data is not public | absolute accuracies are optimistic for every head; the comparisons between heads share the same frozen features |
| Tsinghua Dogs re-uses Stanford Dogs photographs (about 18,000, by its own account) | measured image by image by `scripts/cross_overlap.py`; fixes the provenance subsets and the transfer test list |

The copies are re-encoded at another resolution, so byte hashes cannot find them. `cross_overlap.py` compares
every Tsinghua image with every Stanford image on two independent signals: the Hamming distance of the 64-bit
pHashes in the manifests, and the cosine similarity of the frozen global descriptors g. A pair is *confirmed*
when both agree (pHash ≤ 8 and cosine ≥ 0.85; `--phash-dist`, `--cos-pair`) or when the cosine alone is
overwhelming (≥ 0.97; `--cos-alone`); a *candidate* is any pair with pHash ≤ 12 or cosine ≥ 0.80. The script also
writes a threshold-sensitivity table and montages of random pairs per similarity band; the montages were checked
by eye (research log). The run recorded in the research log found 19,081 confirmed pairs covering 18,675 Tsinghua
and 18,578 Stanford images, and 3,115 of the 5,200 Tsinghua test images with a confirmed Stanford copy.

**Stanford Dogs and ILSVRC-2012, by file id.** Stanford Dogs keeps the ImageNet file names, so
`scripts/ilsvrc_overlap.py` checks each image against the official ILSVRC-2012 training list (`train.txt` of BVLC
Caffe's `caffe_ilsvrc12.tar.gz`, 1,281,167 lines; usage in the script). Result (`results/audit/ilsvrc_overlap.json`):
all 120 breeds are ILSVRC-1k classes, and **every Stanford Dogs image is an ILSVRC-2012 training image**,
12,000 of 12,000 training and 8,580 of 8,580 test images. Through the confirmed copies, so are all 3,115
Stanford-origin Tsinghua test images (59.9 % of the 5,200; a lower bound, since the confirmed set is one).

**Why the provenance subsets exist.** That exposure makes absolute accuracy on the Tsinghua test split optimistic
for every head, and one head could profit from memorised images more than another. `scripts/subset_eval.py`
therefore repeats the per-head test top-1 and the paired comparisons of the main evaluation on two subsets of
the test split, from the saved test predictions; nothing is trained or selected on them. The subsets are fixed
by the overlap audit alone (`results/audit/cross_pairs.csv`), before any test prediction exists. As recorded in
`research_log/research_log.md`:

| Subset | Definition | Images |
|---|---|---|
| `stanford_origin` | a confirmed copy | 3,115 |
| `independent` | no candidate pair at all | 1,362 |
| `ambiguous` | the remaining (candidate only); counted but not analysed | 723 |

The confirmed set is a lower bound on the Stanford-origin images (pHash is not crop-invariant), so the
independent subset excludes every candidate, not only the confirmed pairs. It is not a random sample: it has
fewer typical Stanford-like photographs and includes the 10 Tsinghua breeds with no confirmed Stanford copy. Its
absolute accuracy is not comparable to the full test set; only the comparisons between heads on it are.

**The transfer test list.** The Stanford test images with no candidate match in Tsinghua train or val
(`results/audit/stanford_clean_test.json`; 973 in the recorded run) are the test set of `transfer_eval.py`.
Heads trained on Tsinghua Dogs label them through the empirical breed correspondence of the confirmed pairs; a
Stanford breed is used only if exactly one Tsinghua breed maps to it, with purity ≥ 0.9 and at least 5 pairs
(`--min-purity`, `--min-pairs`).

## Honesty rules the code enforces

* **Selection on validation only.** No training or selection script loads a test split: `run_heads.py` loads
  train and val, and `run_experiment.sh` calls nothing else. Each head's learning rate is chosen on seed 0 by
  validation top-1 (ties: lower validation NLL); early stopping and the kept epoch use the same key. Ablation,
  open-set and token-width runs borrow the rate of the head they modify in `tsinghua_main`. Temperatures are
  fitted on each run's validation predictions, and the hard pairs are the linear head's most-confused pairs on
  validation. Where a hypothesis speaks of the "best" proposed head or control, `make_tables.py` picks it on
  Tsinghua validation top-1 (mean over seeds; ties: lower validation NLL), and Stanford repeats the same pair.
* **The test split is touched only after the freeze.** `final_eval.py`, `subset_eval.py`, `ood_eval.py`,
  `transfer_eval.py` and `explain.py` call `require_freeze` (`src/dogbreed/evaluation/runs.py`) and stop if the
  experiment has no `freeze.json`, if a run is not listed in it (trained after the freeze), or if a run's
  `best.pt` no longer has the SHA-256 recorded at the freeze. A second freeze needs `--force` and an entry in the
  research log. `final_all.sh` fixes the views, the comparison pairs and their order in advance, and the
  provenance subsets are fixed by the overlap audit before any test prediction exists.
* **Annotation-assisted views are flagged.** `head_occlude`, `bg_gray`, `bg_blur`, `crop_body` and `crop_head`
  use the datasets' ground-truth boxes. They are evaluated in their own pass, reported in their own columns
  ("Annotation-assisted (ground-truth boxes)") and rows of the robustness table and figure, left out of the
  robustness hypothesis H7 (automatic corruptions only), and never reported as automatic inference. `explain.py`
  uses the boxes only to measure where the attention falls.
* **Paired comparisons, a fixed rule.** Every comparison is computed per seed on the same test images: the
  accuracy difference, its paired-bootstrap 95 % CI (10,000 resamples of the test images) and the exact McNemar
  test. "A > B" needs the difference to be positive in all three seeds and its CI to exclude zero in at least two
  (`docs/design.md` §8, applied by `make_tables.py`). The `random` control is evaluated with draws fixed by the
  run's seed, and the spread of its top-1 over five draws is reported (`top1_draw_sd`).
* **The paper's numbers come from the digest.** `make_tables.py` reads only the digest, whose files are listed
  with their SHA-256 in `MANIFEST.txt`, and `make_figures.py` only what `make_tables.py` wrote (plus, for the
  attention figure, the saved maps and the images).

## Results and status

*State on 2026-11-09, 04:50 UTC.* Every freeze and evaluation is in
[`research_log/research_log.md`](research_log/research_log.md) with its time.

| Experiment | State |
|---|---|
| `stanford_main`, `tsinghua_main` | **done**: frozen, then test-evaluated once (20:13 / 21:03 UTC freezes) |
| `tsinghua_resnet50` | **done**: frozen by the watcher at 21:56:00 UTC, then test-evaluated |
| `tsinghua_ablate` (K sweep = H6, training choices) | **done on Kaggle**: interrupted on Colab at 22 of 39 runs; the resume rule kept them (largest validation change 0.0003, limit 0.001); the other 17 trained on Kaggle; frozen 03:06:03 UTC, then test-evaluated |
| `tsinghua_convnext_tiny` | **done on Kaggle**: 6 Colab runs kept by the same rule (change 0.0000); frozen 02:40:25 UTC, then test-evaluated |
| `tsinghua_openset`, `tsinghua_clip_vitb16`, `tsinghua_tok32`, `tsinghua_tok64`, efficiency report | **done on Kaggle** (02:09–03:54 UTC): each experiment frozen, then test-evaluated once; the efficiency report measured last, on a Tesla T4 |
| failed runs | none |

The Kaggle results are on Kaggle as two archives and are not yet merged into the digest, the tables or the paper
(`scripts/ingest_kaggle.sh`); the research log quotes the test top-1 its logs print. The numbers below come from the
locked test evaluations: heads on frozen DINOv2 ViT-B/14-reg features, three seeds, mean ± s.d. The paper
([`paper/main.pdf`](paper/main.pdf)) adds the confidence intervals, the tests, calibration and the per-breed results.

**Test top-1 (%)**

| Head | Params (Tsinghua) | Tsinghua Dogs (5,200 test images) | Stanford Dogs (8,580) |
|---|---|---|---|
| B0 linear probe | 202,882 | 86.26 ± 0.21 | 88.33 ± 0.07 |
| B1 attentive probe | 404,866 | 86.26 ± 0.02 | 86.71 ± 0.49 |
| B2 class attention, all classes | 302,211 | 86.28 ± 0.29 | 88.34 ± 0.08 |
| Re-rank, class query (control) | 302,211 | 86.71 ± 0.10 | 88.28 ± 0.05 |
| Re-rank, centred query (control) | 302,211 | 86.49 ± 0.17 | 88.26 ± 0.09 |
| Re-rank, random rivals (control) | 302,211 | 86.43 ± 0.21 | 88.16 ± 0.10 |
| RCEA rival | 302,211 | 86.74 ± 0.23 | 88.17 ± 0.12 |
| RCEA rival-soft | 302,211 | **87.12 ± 0.15** | 88.25 ± 0.19 |
| RCEA pairwise | 302,211 | 86.80 ± 0.12 | 88.18 ± 0.12 |

**The paper's question gets a negative answer: H1 is refuted.** The proposed head picked on Tsinghua validation
is rival-soft. On Tsinghua it is +0.41 pp ahead of the best same-size control, the class query. The difference is
positive in each seed (+0.37, +0.50, +0.37), but the paired 95 % CI clears zero in only one seed, and the
pre-registered rule needs two. On Stanford the same pair is level (−0.03 pp). So conditioning the query on the
rivals is not shown to matter. What does hold, on Tsinghua only:

* **Re-ranking a top-5 shortlist helps** (H5 supported). Rival-soft beats B0 by +0.86 pp and all-class attention
  B2 by +0.84 pp, both by the rule. It does so on the 1,362 test images with no Stanford copy too (+1.32 and
  +1.39 pp).
* **The gain sits where it should** (H2 supported). The re-ranker fixes 1.91 % of the test images and breaks
  0.68 %. It gains +1.34 pp on the 20 most-confused breed pairs, against +0.86 pp overall.
* **Random rivals do not carry it** (H3 supported by its pre-registered criterion): random 86.43 < centred
  86.49 < rival 86.74. The paired rule itself makes no claim for rival over random (+0.31 pp); it does for
  rival-soft over random (+0.69 pp).

On Stanford, where B0 already reaches 88.3 %, no re-ranker is ahead of B0 (rival-soft −0.09 pp). H2 and H3 are
refuted there, and H5 is not. Pairwise against rival shows no difference (H4, exploratory).

* **Corruptions** (H7, exploratory, automatic corruptions only). On Tsinghua, rival-soft's lead over B0 grows
  under blur (+3.03 pp), occlusion (+2.64) and low resolution (+1.73), and stays near the clean value under noise
  (+0.83) and JPEG (+0.76). On Stanford it is −0.07 to −0.23 pp throughout.
* **Transfer.** Heads trained on Tsinghua were tested on the 973 Stanford test images with no copy in Tsinghua
  train or val. Top-1: rival-soft 79.75 ± 0.54, class query 79.00 ± 0.43, B0 78.14 ± 0.16.
* **Where the attention goes** (ground-truth boxes used for this measurement only, seed 0). The class query
  concentrates on the dog's head: head-box mass 51.7 % against 29.4 % of the area, normalised entropy 0.79. The
  rival queries spread almost uniformly: head-box mass 26.8–32.1 %, entropy 0.96–0.98.
* **Exposure.** Every Stanford Dogs image is an ILSVRC-2012 training image ([above](#the-leakage-and-provenance-audit)).
  Absolute accuracies are optimistic for every backbone here; the comparisons between heads share the same
  features.

* **Other backbone.** With frozen ResNet-50 features the heads are 0.7–1.0 pp lower: B0 85.59 ± 0.11, class
  query 85.76 ± 0.01, rival 85.85 ± 0.05, pairwise 85.79 ± 0.03. Rival minus class is +0.09 pp, no claim.

| Not yet in | State |
|---|---|
| K ablation (H6), training-choice ablations, open set, token width 32 / 64, ConvNeXt-T, CLIP ViT-B/16, efficiency | done on Kaggle; not yet merged into the digest and the paper |
| End-to-end fine-tuning | not run on the free tier; the study measures heads on frozen features only (`docs/design.md` §11) |
| Learning-rate grid | every head chose 3e-4, the low edge of {3e-4, 1e-3, 3e-3}; no lower rate was tried |

The raw results are on the Drive mirror of the runtime's work folder ([Outputs](#outputs)). The digest, with its
SHA-256 manifest, is in [`results/digest/`](results/digest/). The paper's tables, quoted numbers and figures are
generated from it into `paper/generated/` and `paper/figures/`. The runtime's full `pip freeze` is
[`requirements-colab-full.txt`](requirements-colab-full.txt), and the SHA-256 of the Tsinghua archives the runs used
are in [`configs/expected_sha256_tsinghua.txt`](configs/expected_sha256_tsinghua.txt).

## Licence

Code: MIT ([`LICENSE`](LICENSE)). The datasets and the pretrained weights keep their owners' terms (above); none
of them is redistributed here. The paper's text and figures are the authors'.
