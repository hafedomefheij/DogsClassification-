"""Write the three Colab notebooks (plain JSON, no nbformat needed).

    python -I notebooks/make_notebooks.py

The notebooks only drive the scripts in scripts/; every number they print comes from those scripts. They
follow the order the study is run in:
  01  data, within-dataset audits, feature caches and the cross-dataset overlap audit;
  02  training (tsinghua_main, then queues A, B and C) and the selection freeze (scripts/freeze.py);
  03  the locked test evaluation (scripts/final_all.sh), efficiency, digest, tables, figures, the paper.
Long jobs are started detached (setsid nohup bash -c '...' > log 2>&1 < /dev/null &), so their cell returns
at once: a cell running `!nohup cmd &` stays busy until the job ends, because the job holds the cell's
terminal. One point needs a person: each freeze is recorded in research_log/research_log.md before
notebook 3 reads that experiment's test split.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells: list) -> dict:
    # nbformat 4.5 wants a unique id per cell; shared cells (SETUP, HELPERS) are copied, not mutated
    cells = [dict(c, id=f"cell-{i:02d}") for i, c in enumerate(cells)]
    return {"cells": cells, "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


SETUP = [
    md("""
## Setup (run at the start of every session)

Runtime → Change runtime type → **T4 GPU** (any CUDA GPU works; the study uses a free-tier T4, 15 GB).
Mounting Google Drive is optional but strongly advised: `scripts/colab_sync.sh` mirrors everything under
`/content/work` (manifests, audits, runs, freezes, results, logs, queue state) to Drive, so a recycled runtime
loses only the feature caches. Notebook 1 rebuilds those from the images. `push()` (below) first copies every
token PCA (`pca.npz`) into `/content/work/pca_backup`, and this cell puts it back, so rebuilt caches use the
basis the heads were trained in (`extract_features.py` reuses an existing `pca.npz` instead of fitting a new one).

**Long jobs run detached.** A cell `!nohup cmd &` stays busy, and keeps the kernel busy, until the job ends,
because the job holds the cell's terminal. Every long job here is started as
`!setsid nohup bash -c '…' > /content/work/logs/<name>.log 2>&1 < /dev/null &`: the cell returns at once, and a
watch cell reads the log. The same commands can be typed into Colab's Terminal panel (without the `!`) after
exporting the variables `scripts/colab_env.py` sets, since a terminal does not have the kernel's environment.
"""),
    code("""
from google.colab import drive
drive.mount('/content/drive')   # optional: skip this cell to keep everything on the runtime's disk
"""),
    code("""
# Where the code comes from: a git URL, or a zip of this repository (top folder DogBreed-PhD-Research)
REPO_URL = ""                                                   # e.g. the URL of your fork
REPO_ZIP = "/content/drive/MyDrive/DogBreed-PhD-Research.zip"
import os, subprocess
D = "/content/DogBreed-PhD-Research"                            # the queue files cd here: keep this path
if not os.path.isdir(D):
    if REPO_URL:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, D], check=True)
    elif os.path.isfile(REPO_ZIP):
        subprocess.run(["unzip", "-q", REPO_ZIP, "-d", "/content"], check=True)
    else:
        raise SystemExit("set REPO_URL, or put the repository zip at REPO_ZIP")
os.chdir(D)
exec(open("scripts/colab_env.py").read())                     # paths: caches on local disk, the rest on Drive
!pip install -q --no-deps -e . && python -m pytest -q tests 2>&1 | tail -2
!bash scripts/colab_sync.sh pull                                # restore earlier sessions' work, if any
!rsync -a /content/work/pca_backup/ /content/features/ 2>/dev/null && find /content/features -name pca.npz
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader


def push():
    \"\"\"Copy every token PCA of the caches into /content/work/pca_backup, then mirror /content/work to Drive.\"\"\"
    !rsync -am --include='*/' --include='pca.npz' --include='pca.json' --exclude='*' /content/features/ /content/work/pca_backup/
    !bash scripts/colab_sync.sh push
"""),
]

HELPERS = [
    md("## Helpers"),
    code("""
import glob, json
import yaml
E, R, W = os.environ["DOGBREED_EXPERIMENTS"], os.environ["DOGBREED_RESULTS"], os.environ["WORK"]
EXPS = ["tsinghua_main", "stanford_main", "tsinghua_ablate", "tsinghua_openset", "tsinghua_resnet50",
        "tsinghua_convnext_tiny", "tsinghua_clip_vitb16", "tsinghua_tok32", "tsinghua_tok64"]


def complete(exp):
    \"\"\"Every run of the experiment's config has finished for every seed: best.pt and summary.json, the two
    files scripts/freeze.py reads (train_heads.py writes best.pt just before summary.json).\"\"\"
    cfg = yaml.safe_load(open(f"configs/experiments/{exp}.yaml"))
    return all(os.path.exists(f"{E}/{exp}/{r['tag']}_s{s}/{f}") for r in cfg["runs"] for s in cfg["seeds"]
               for f in ("best.pt", "summary.json"))


def frozen(exp):
    \"\"\"The experiment's freeze.json (scripts/freeze.py), or None.\"\"\"
    f = f"{E}/{exp}/freeze.json"
    return json.load(open(f)) if os.path.exists(f) else None


def final_done(exp):
    \"\"\"scripts/final_all.sh finished this experiment (its log ends with FINAL DONE).\"\"\"
    f = f"{W}/logs/final_{exp}.log"
    return os.path.exists(f) and f"FINAL DONE {exp}" in open(f).read()


def running(pattern):
    \"\"\"A process whose command line matches (pgrep -f); write "[x]yz" so a shell never matches itself.\"\"\"
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0
"""),
]


NB1 = notebook([
    md("""
# 1 · Data, audits and feature caches

Downloads Tsinghua Dogs and Stanford Dogs from their owners, verifies them, builds the manifests and the
within-dataset audits, caches the frozen-backbone features every later step reads, and runs the cross-dataset
overlap audit that fixes the provenance subsets of the Tsinghua test split. Nothing here trains a model or
evaluates anything on a test split.

* **Licences.** Tsinghua Dogs: academic use, citation requested. Stanford Dogs: ImageNet terms
  (non-commercial research and education, no redistribution). Nothing here redistributes images.
* **Stanford Dogs source.** If vision.stanford.edu does not answer, `scripts/download_data.sh` falls back
  to Internet Archive snapshots of the official URLs. It accepts a file only if its SHA-256 equals the
  checksum TensorFlow Datasets records for the official release.
* **Splits.** The validation split is 10 % of the official training images (stratified, seed 0).
  Training images that duplicate a test image are excluded. The test split is the official one (for
  Tsinghua Dogs, the official validation list).
"""),
    *SETUP,
    md("""
## 1. Download, verify, manifests, within-dataset audits

`scripts/colab_data.sh all` copies the archives already saved on Drive, downloads the rest
(`scripts/download_data.sh`: resumed downloads, SHA-256 records), saves new archives to Drive, unpacks them and
runs `scripts/prepare_data.py` for each dataset. That writes the manifest (one row per image: label, official
split, our split, decode status, hashes) and the audit: exact and near duplicates (SHA-1, pHash, dHash),
training images that duplicate a test image (excluded from training), near pairs with different labels.
The log ends with `DATA DONE (all)`.
"""),
    code("""
!setsid nohup bash -c 'bash scripts/colab_data.sh all' > /content/work/logs/data.log 2>&1 < /dev/null &
"""),
    code("""
!tail -n 4 /content/work/logs/data.log
"""),
    md("When the log ends with `DATA DONE (all)`:"),
    code("""
!head -n 60 $DOGBREED_RESULTS/audit/tsinghua.json
push()
"""),
    md("""
## 2. Feature caches (DINOv2 ViT-B/14 with registers, 224 px, frozen)

`scripts/extract_all.sh <dataset> <backbone> <images> [token_pca] [img_size]`:
* Global descriptor: g = [CLS ; mean patch], 1536-d.
* Tokens: the 256 patch tokens, projected on a 256-component PCA. The PCA is fitted on 4,000 *training*
  images only.
* Views:
  * orig: train, val and test;
  * hflip: train only;
  * five automatic corruptions of the test split (blur, noise, jpeg, lowres, occlude);
  * five annotation-assisted test views (head_occlude, bg_gray, bg_blur, crop_body, crop_head). These use the
    dataset's boxes, so they are robustness probes, never automatic inference.

Every cache resumes after an interruption: start the cell again. The log ends with
`EXTRACT DONE tsinghua dinov2_vitb14_reg`. Training (notebook 2) reads the orig and hflip caches; the test views
are read only by the locked evaluation (notebook 3).
"""),
    code("""
!setsid nohup bash -c 'bash scripts/extract_all.sh tsinghua dinov2_vitb14_reg /content/data/raw/tsinghua_dogs/low-resolution.zip 256 224' > /content/work/logs/extract_tsinghua.log 2>&1 < /dev/null &
"""),
    code("""
!tail -n 3 /content/work/logs/extract_tsinghua.log; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
"""),
    md("When it is done:"),
    code("""
push()
"""),
    md("""
## 3. Stanford Dogs caches and the cross-dataset audit

Run after the Tsinghua caches (the transfer cache reuses the Tsinghua PCA). `scripts/pipeline_stanford.sh`:
1. the Stanford caches in Stanford's own PCA basis (orig, hflip, the automatic test views, and the
   annotation-assisted body-box views bg_gray, bg_blur, crop_body: Stanford Dogs has no head boxes);
2. the Stanford test split in the Tsinghua PCA basis (feature tag `pca256_tsinghua`), for the transfer study;
3. `scripts/cross_overlap.py`: the Stanford Dogs photographs inside Tsinghua Dogs, from two signals (pHash
   distance and the cosine of the frozen descriptors g), with a threshold-sensitivity table and montages per
   similarity band for a visual check. It also writes the de-duplicated Stanford test list for the transfer study.

The log ends with `STANFORD PIPELINE DONE`. Queue B (notebook 2) waits for this script to end before it trains
`stanford_main`, so notebook 2 can be started as soon as the Tsinghua caches are done.
"""),
    code("""
!setsid nohup bash -c 'bash scripts/pipeline_stanford.sh' > /content/work/logs/pipeline_stanford.log 2>&1 < /dev/null &
"""),
    code("""
!tail -n 3 /content/work/logs/pipeline_stanford.log
"""),
    md("""
If `cross_overlap.py` runs out of GPU memory (training jobs share the GPU), run the audit alone on the CPU: the
same computation, only slower.
"""),
    code("""
!setsid nohup bash -c 'python -I scripts/cross_overlap.py --backbone dinov2_vitb14_reg --feature-tag pca256 --tsinghua-images /content/data/raw/tsinghua_dogs/low-resolution.zip --stanford-images /content/data/raw/stanford_dogs/Images --device cpu' > /content/work/logs/cross_overlap.log 2>&1 < /dev/null &
"""),
    md("""
## 4. The audit's numbers and the provenance subsets

The provenance subsets of the Tsinghua test split come from `results/audit/cross_pairs.csv` alone, before any
test prediction exists (the rule is `provenance_subsets` in `scripts/subset_eval.py`):
* **stanford_origin**: a confirmed Stanford Dogs copy;
* **independent**: no candidate pair at all;
* **ambiguous**: a candidate but no confirmed pair (counted, not analysed).

The run recorded in `research_log/research_log.md` gave 3,115 / 1,362 / 723 of the 5,200 test images. From
here on, keep `results/audit/` as it is: it defines the subsets.
"""),
    code("""
import importlib.util, json
import pandas as pd
R, M = os.environ["DOGBREED_RESULTS"], os.environ["DOGBREED_MANIFESTS"]
r = json.load(open(f"{R}/audit/cross_tsinghua_stanford.json"))
print({k: r[k] for k in ("candidate_pairs", "confirmed_pairs", "confirmed_tsinghua_images",
                         "confirmed_stanford_images", "stanford_test_in_tsinghua_trainval", "stanford_test_clean")})
spec = importlib.util.spec_from_file_location("subset_eval", "scripts/subset_eval.py")
se = importlib.util.module_from_spec(spec)
spec.loader.exec_module(se)
m = pd.read_csv(f"{M}/tsinghua.csv", usecols=["id", "split"])
subs = se.provenance_subsets(pd.read_csv(f"{R}/audit/cross_pairs.csv", usecols=["t_id", "confirmed"]),
                             m.loc[m["split"] == "test", "id"])
print("Tsinghua test split by provenance:", {k: len(v) for k, v in subs.items()})
push()
"""),
    md("""
## 5. Other backbones

The Tsinghua caches of ResNet-50, ConvNeXt-T and CLIP ViT-B/16 (orig and hflip only: they enter the comparison,
not the robustness study) are extracted by queue B in notebook 2 (jobs `extract_resnet50`,
`extract_convnext_tiny`, `extract_clip_vitb16`); queue A's backbone runs wait for the last of them.

## After a recycled runtime: rebuild the caches

The setup cell brings `/content/work` back from Drive, and the token PCAs with it; the feature caches are gone.
This cell rebuilds every cache the later steps read. It reuses the archives on Drive and the restored manifests
(`colab_data.sh all --no-prepare`) and the restored PCAs, and it does **not** re-run the overlap audit, whose
`cross_pairs.csv` defines the provenance subsets. Queue B does not redo extraction jobs it has marked done, so
the other backbones are rebuilt here too. The log ends with the last `EXTRACT DONE` line.
"""),
    code("""
R = os.environ["DOGBREED_RESULTS"]
assert os.path.exists(f"{R}/audit/cross_pairs.csv"), "no overlap audit yet: run sections 1-3 instead"
T, S = "/content/data/raw/tsinghua_dogs/low-resolution.zip", "/content/data/raw/stanford_dogs/Images"
steps = ["bash scripts/colab_data.sh all --no-prepare",
         f"bash scripts/extract_all.sh tsinghua dinov2_vitb14_reg {T} 256 224",
         f"ANN_VIEWS=bg_gray,bg_blur,crop_body bash scripts/extract_all.sh stanford dinov2_vitb14_reg {S} 256 224",
         f"python -I scripts/extract_features.py --dataset stanford --backbone dinov2_vitb14_reg --images {S} "
         "--splits test --views orig --token-pca 256 --pca-from tsinghua --workers 2"]
steps += [f"AUTO_VIEWS= ANN_VIEWS= bash scripts/extract_all.sh tsinghua {b} {T} 256 224"
          for b in ("resnet50", "convnext_tiny", "clip_vitb16")]
chain = " && ".join(steps)
!setsid nohup bash -c '{chain}' > /content/work/logs/rebuild_caches.log 2>&1 < /dev/null &
"""),
    code("""
!tail -n 3 /content/work/logs/rebuild_caches.log
"""),
])


NB2 = notebook([
    md("""
# 2 · Training, model selection and the freeze

Every head is trained on the cached features; the backbone never changes.

* For each head, the learning rate in {3e-4, 1e-3, 3e-3} with the best **validation** top-1 on seed 0 is
  kept (ties: lower validation NLL; `experiments/<exp>/lr_selection.json`). The ablation, open-set and
  token-width runs borrow the rate selected for their head in `tsinghua_main` (`lr_selection_from`, `lr_like`)
  and select nothing themselves; each other backbone has its own grid.
* Seeds 0, 1 and 2 are then trained at that rate, for at most 30 epochs, with early stopping on validation
  top-1 (patience 8).
* No cell in this notebook reads a test split. It ends with the selection freeze (`scripts/freeze.py`).

Heads, at equal parameter counts for all re-rankers (302,211 on Tsinghua Dogs):

| Group | Heads |
|---|---|
| Baselines | `linear`, `attnprobe`, `classattn_all` |
| Controls (same size, rival-blind queries) | `rr_class`, `rr_center`, `rr_random` |
| Proposed (rival-conditioned queries) | `rr_rival`, `rr_rival_soft`, `rr_pairwise` |
"""),
    *SETUP,
    *HELPERS,
    md("""
## The main experiment: `tsinghua_main` (about 1.5–3 h on a T4)

`scripts/run_experiment.sh` runs the learning-rate grid on seed 0 (`run_heads.py --select-lr`), then seeds 0–2
at the chosen rates (`run_heads.py`). Finished runs are skipped; an interrupted run resumes from its `last.pt`.
The log ends with `TRAINING DONE configs/experiments/tsinghua_main.yaml`. Start it exactly as below: queue A
recognises it by this command line.
"""),
    code("""
if running("[r]un_experiment.sh configs/experiments/tsinghua_main.yaml"):
    print("tsinghua_main is already running")
else:
    !setsid nohup bash -c 'bash scripts/run_experiment.sh configs/experiments/tsinghua_main.yaml' > /content/work/logs/tsinghua_main.log 2>&1 < /dev/null &
"""),
    md("""
## Everything else, unattended: queues A, B and C

`scripts/queue.sh` runs the `name | command` lines of a queue file in order, each once. A finished job leaves
`<name>.done`, a failed one `<name>.failed` (and the queue moves on), in `/content/work/logs/queue_<a|b|c>/`; its
output goes to `/content/work/logs/<name>.log`. Lines added while the queue runs are picked up; the queue exits
once every job is marked and the file has an `END` line.

| Queue | Jobs, in order |
|---|---|
| A, `configs/queues/a.txt`: Tsinghua-sized jobs (about 9 GB of tokens on the GPU each), one at a time | `wait_main` (until `tsinghua_main` has finished) → `ablate` (`tsinghua_ablate`) → `openset_split` (`make_openset_split.py --dataset tsinghua --unknown 30 --seed 0`) → `openset` (`tsinghua_openset`) → `wait_bb` (until queue B's `extract_clip_vitb16` is marked) → `tok32`, `tok64` |
| B, `configs/queues/b.txt`: Stanford-sized jobs, alongside A | `wait_stanford` (until `pipeline_stanford.sh` has finished) → `stanford_main` → `extract_resnet50`, `extract_convnext_tiny`, `extract_clip_vitb16` (Tsinghua caches, orig and hflip) |
| C, `configs/queues/c.txt`: the other backbones, alongside A | `wait_b` (until queue B's `extract_clip_vitb16` is marked) → `bb_resnet50`, `bb_convnext_tiny`, `bb_clip_vitb16` (each with its own learning-rate grid) |

Start the queues only once `tsinghua_main` is running (or complete): otherwise `wait_main` passes at once and
the ablations fail for want of its learning rates. A queue that is already running is not started twice.
"""),
    code("""
assert running("[r]un_experiment.sh configs/experiments/tsinghua_main.yaml") or complete("tsinghua_main"), \\
    "start tsinghua_main first (cell above)"
for q in ("a", "b", "c"):
    if running(f"[q]ueue.sh configs/queues/{q}.txt"):
        print(f"queue {q} is already running")
    else:
        !setsid nohup bash -c 'bash scripts/queue.sh configs/queues/{q}.txt' > /content/work/logs/queue_{q}.log 2>&1 < /dev/null &
"""),
    md("## Watch"),
    code("""
!tail -n 2 /content/work/logs/tsinghua_main.log /content/work/logs/queue_a.log /content/work/logs/queue_b.log /content/work/logs/queue_c.log
!ls /content/work/logs/queue_a /content/work/logs/queue_b /content/work/logs/queue_c 2>/dev/null; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
push()
"""),
    md("""
A failed job's output is in `/content/work/logs/<job>.log`. To run it again, delete its `.failed` marker: a
running queue picks it up, a queue that has exited must be started again (cell above).

## After a recycled runtime

Run the setup cells and the last section of notebook 1 (the caches), then the two launch cells above again:
finished runs are skipped and jobs marked done are not repeated. A run cut off mid-way starts again from its
first epoch, since `last.pt` is not mirrored to Drive. The open-set split is written into the repository
checkout (`configs/openset/`), which is not mirrored either; write it again (same seed, same breeds) before the
open-set runs resume:
"""),
    code("""
![ -f configs/openset/tsinghua_unknown30_s0.json ] || python -I scripts/make_openset_split.py --dataset tsinghua --unknown 30 --seed 0
"""),
    md("## Validation results (no test data)"),
    code("""
import pandas as pd
for f in sorted(glob.glob(f"{E}/*/lr_selection.json")):
    print(f.split("/")[-2], json.load(open(f)))
rows = []
for f in sorted(glob.glob(f"{E}/*/*_s[0-9]/summary.json")):
    exp, run = f.split("/")[-3:-1]
    if "_lr" in run:                                   # learning-rate grid runs
        continue
    r = json.load(open(f))
    rows.append({"experiment": exp, "run": run, "lr": r["config"].get("lr"), "val_top1": r["val_top1"],
                 "val_nll": r["val_nll"], "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
                 "params": r["params"]})
print(pd.DataFrame(rows).to_string(index=False))
"""),
    md("""
## The selection freeze — stop here

When every configured run of an experiment has finished, `scripts/freeze.py --experiment <exp>` writes
`experiments/<exp>/freeze.json`: the time (UTC), each head's learning rate, and for every seed run its
validation metrics, best epoch and the SHA-256 of its `best.pt`. Every script that reads a test split refuses
an experiment without it, a run it does not list, and a checkpoint whose hash has changed since
(`require_freeze` in `src/dogbreed/evaluation/runs.py`). `freeze.py` refuses a head with fewer than
`--expect-seeds` (3) finished seeds, and a second freeze without `--force` (a protocol event that belongs in the
research log). It only sees heads with at least one finished seed, so this cell freezes an experiment only when
`complete()` finds every run of its config.

Paste each printed summary, with its time, into the section "Model selection freeze" of
`research_log/research_log.md`. Only then open notebook 3 for that experiment.
"""),
    code("""
for exp in EXPS:
    fz = frozen(exp)
    if fz:
        print(f"{exp}: frozen {fz['frozen_at_utc']} UTC, {len(fz['runs'])} runs")
    elif not complete(exp):
        print(f"{exp}: training not complete, not frozen")
    else:
        !python -I scripts/freeze.py --experiment {exp}
push()
"""),
])


NB3 = notebook([
    md("""
# 3 · Locked test evaluation and report (after the selection freeze)

This notebook reads the test splits. Run it for an experiment only after notebook 2 has frozen its model
selection and the freeze is recorded in `research_log/research_log.md`; every script that reads a test split
refuses an experiment without a matching `freeze.json`.

* `scripts/final_all.sh <exp>` is the whole locked evaluation of one experiment. Its views, comparison pairs
  and their order are fixed in the script (`docs/design.md` §8–9), so nothing is chosen after a test result.
  Temperatures are fitted on validation predictions, the hard pairs come from the linear head's validation
  confusions, and the "best" proposed head and control of the tables are picked on validation top-1.
* The annotation-assisted views (head_occlude, bg_gray, bg_blur, crop_body, crop_head) use the datasets' boxes.
  They are evaluated in their own pass, reported in their own columns, and never as automatic inference.
* Then: efficiency, the digest, the tables, the figures and the paper.
"""),
    *SETUP,
    *HELPERS,
    md("## Which experiments are frozen"),
    code("""
for exp in EXPS:
    fz = frozen(exp)
    print(f"{exp:<24}", f"frozen {fz['frozen_at_utc']} UTC, {len(fz['runs'])} runs" if fz else "NOT FROZEN",
          "| locked evaluation done" if final_done(exp) else "")
"""),
    md("""
The open-set split lives in the repository checkout (`configs/openset/`), which `colab_sync.sh` does not mirror.
After a recycled runtime, write it again (same seed, same breeds) before `tsinghua_openset` is evaluated:
"""),
    code("""
![ -f configs/openset/tsinghua_unknown30_s0.json ] || python -I scripts/make_openset_split.py --dataset tsinghua --unknown 30 --seed 0
"""),
    md("""
## The locked evaluation (detached, one experiment after another)

| Experiment | What `scripts/final_all.sh` runs, in order |
|---|---|
| `tsinghua_main` | `final_eval.py` on orig with the 22 main pairs; on the automatic corruptions (blur, noise, jpeg, lowres, occlude); separately on the annotation-assisted views (head_occlude, bg_gray, bg_blur, crop_body, crop_head), both with 7 pairs against B0 and the class query; `subset_eval.py` (the provenance subsets, orig, the 22 main pairs); `transfer_eval.py` (Tsinghua → clean Stanford test); `explain.py` (attention maps and box statistics) |
| `stanford_main` | the same three `final_eval.py` passes, with bg_gray, bg_blur, crop_body as the annotation-assisted views |
| `tsinghua_ablate` | `final_eval.py` on orig: rival vs class query at K = 2, 3, 10; pairwise vs rival at K = 3, 10 |
| `tsinghua_openset` | `ood_eval.py`: known vs held-out breeds (AUROC, FPR at 95 % TPR, closed-set top-1) |
| `tsinghua_resnet50`, `tsinghua_convnext_tiny`, `tsinghua_clip_vitb16` | `final_eval.py` on orig: rival and pairwise vs class query, pairwise vs rival, rival and class query vs B0 |
| `tsinghua_tok32`, `tsinghua_tok64` | `final_eval.py` on orig: rival vs class query, rival and class query vs the attentive probe |

Each comparison is computed per seed on the same test images (paired bootstrap, 10,000 resamples; exact
McNemar test). This cell starts every frozen experiment that has no finished evaluation yet, one after another;
each writes `/content/work/logs/final_<exp>.log`, ending with `FINAL DONE <exp>`. A failed evaluation is run
again whole the next time; record why in the research log.
"""),
    code("""
todo = [e for e in EXPS if frozen(e) and not final_done(e)]
chain = " ; ".join(f"bash scripts/final_all.sh {e} > /content/work/logs/final_{e}.log 2>&1" for e in todo)
if not todo:
    print("nothing to start: no frozen experiment is waiting for its locked evaluation")
elif running("[f]inal_all.sh"):
    print("a locked evaluation is still running: start this cell again when it has finished")
else:
    print("locked evaluation of:", ", ".join(todo))
    !setsid nohup bash -c '{chain}' > /content/work/logs/final_all.log 2>&1 < /dev/null &
"""),
    code("""
!tail -n 2 /content/work/logs/final_*.log
"""),
    md("""
### Or: finish the secondary experiments unattended

The recorded run finished the ablation, open-set, backbone and token-width experiments with
`scripts/auto_final.sh` (research log, "Automatic freeze and evaluation of the secondary experiments"). Started
once beside the queues, it freezes each experiment as soon as its queue job is marked done, runs its locked
evaluation, and at the end the efficiency report, digest, tables and figures. It chooses nothing the cells here
would not; the freeze printouts it keeps in `/content/work/logs/freeze_<exp>.log` go into the research log
afterwards.
"""),
    code("""
if running("[a]uto_final.sh"):
    print("auto_final.sh is already running")
else:
    !setsid nohup bash -c 'bash scripts/auto_final.sh' > /content/work/logs/auto_final.log 2>&1 < /dev/null &
"""),
    md("When every experiment's log ends with `FINAL DONE`:"),
    code("""
push()
"""),
    md("""
## Efficiency (with nothing else on the GPU)

`scripts/efficiency_report.py` measures parameters, FLOPs (`torch.utils.flop_counter`; a multiply-add = 2
FLOPs), latency (median of `--iters` timed passes, default 50, after 10 warm-up passes) and peak GPU memory:
every head on random inputs of the cached shapes (batch 1 and 256, fp32), the frozen backbone (batch 1 and 64,
fp16 autocast on a T4), and the full single-image pipeline (backbone → token PCA → head). It reads no test data and
needs no freeze, but its timings only mean something when no other job uses the GPU (the first line lists the
processes on it). Output: `results/efficiency/tsinghua_main.json` and `.csv` (the paper's table reads this
experiment).
"""),
    code("""
!nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
!python -I scripts/efficiency_report.py --config configs/experiments/tsinghua_main.yaml
"""),
    md("""
## Digest

`scripts/digest.py` collects every result into small CSV/JSON files in `$DOGBREED_RESULTS/digest/`, with
`MANIFEST.txt` holding the SHA-256 and size of each; the tables and figures are built from these files only.
`--curves` adds the per-epoch validation curves of the named runs. The curves figure draws seed 0 of B0, the
class query and the proposed head picked on validation, so the list holds all three proposed heads, Tsinghua
runs only.
"""),
    code("""
!python -I scripts/digest.py --curves tsinghua_main/linear_s0,tsinghua_main/rr_class_s0,tsinghua_main/rr_rival_s0,tsinghua_main/rr_rival_soft_s0,tsinghua_main/rr_pairwise_s0
push()
"""),
    md("""
## Tables and figures (from the digest only)

`scripts/make_tables.py` writes `paper/generated/`: `tab_*.tex` (one booktabs tabular per table, input by
`paper/main.tex`), `numbers.json` (the numbers the text quotes, H1–H7 judged by the rule of `docs/design.md` §8,
and the tables skipped for missing inputs) and `figdata.json`. `scripts/make_figures.py` draws `paper/figures/`
(vector PDFs; `--png` adds previews); the attention figure also reads the maps `explain.py` saved and the
original images. Both name what they skipped.
"""),
    code("""
!python -I scripts/make_tables.py --digest $DOGBREED_RESULTS/digest --out paper/generated
!python -I scripts/make_figures.py --gen paper/generated --out paper/figures --explain $DOGBREED_RESULTS/tsinghua_main/explain/shown_maps.json --images /content/data/raw/tsinghua_dogs/low-resolution.zip --png
"""),
    md("""
## The paper (TeX Live from apt, then latexmk)

`paper/refs.bib` is built from the two verified bibliographies by `python -I paper/merge_bib.py`. The last cell
copies the paper's inputs and the PDF into `/content/work/paper`, since `paper/` is in the repository checkout,
which `colab_sync.sh` does not mirror. The digest itself is on Drive under `work/results/digest/`: copy it into
the repository's `results/digest/` and check every file against `MANIFEST.txt` (README, "How to reproduce").
"""),
    code("""
!apt-get -qq update > /dev/null && apt-get -qq install -y latexmk texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended > /dev/null && latexmk -v | head -n 2
"""),
    code("""
!cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > /content/work/logs/latexmk.log 2>&1; tail -n 3 /content/work/logs/latexmk.log; ls -l main.pdf
"""),
    code("""
!mkdir -p /content/work/paper && cp -r paper/generated paper/figures paper/main.pdf /content/work/paper/
push()
"""),
])


def main():
    for name, nb in [("01_data_and_features", NB1), ("02_train_and_select", NB2), ("03_evaluate_and_report", NB3)]:
        p = HERE / f"{name}.ipynb"
        p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(p.name, len(nb["cells"]), "cells")


if __name__ == "__main__":
    main()
