# Research log

One entry per decision or event, in order. Times are UTC (the Colab VM clock; local logs on the Windows PC are
UTC+2 and have been converted). Each entry says what was done, what was observed, and what was decided. Nothing
here is a result unless it says where the number came from.

---

## 2026-11-08

### Scope and constraints (from the brief)
- Goal: a novel deep-learning method for fine-grained dog-breed classification, Tsinghua Dogs primary, Stanford
  Dogs secondary, with a full experimental and reporting protocol.
- Hard rules carried through the project: no fabricated citations, numbers, logs or conclusions; test data
  untouched until model selection is frozen; no test information in training or tuning; annotation-assisted
  results never reported as automatic inference; no paid compute, billing changes or access workarounds.

### Literature and novelty gate
- Wrote `research_log/lit_core.md` (the four named papers read in full, dataset facts and licences, the novelty
  search) and `research_log/lit_related.md` (verified related-work dossier). Table: `results/literature_table.csv`
  (62 rows; 46 fully verified against a primary source, 16 partially, each row says which).
- Novelty verdict (`lit_core.md` §3.5): the rival-conditioned query is **novel as a specific mechanism and
  incremental as an idea**; closest works CSRA (mechanism), SAC (pipeline), FuDD (idea). Decided: cite all three
  up front; untie query embeddings from the classifier; add the equal-parameter controls `class`, `center`,
  `random`; add the pairwise / soft-minimum variant, which is not reducible to centring.
- Candidate designs compared on paper (`docs/design.md` §2): A rival-conditioned evidence attention (selected),
  B slot-based part discovery (rejected: covered by Relational Proxies and PDiscoNet), C confusion-graph margins
  with gated refinement (rejected: two known parts that confound each other), D head–body box-distilled attention
  (kept as a possible auxiliary; needs part boxes Stanford Dogs lacks).

### Compute
- Local PC: Python 3.12.15 venv, torch 2.15.0+cpu, timm 1.0.31, 2 CPU threads (`research_log/env_setup.log`).
  Enough for unit tests and smoke runs only.
- The user asked for Colab instead of the CPU and gave the agent control of a Colab notebook in the app's browser
  pane. Runtime switched to GPU: **NVIDIA T4, 15 GB; 12 GB RAM; 2 vCPU; ~113 GB disk** (`nvidia-smi`, `free`,
  `df` in the status cell). The runtime offered only the T4, i.e. the free tier; no paid resources were bought.
  Google Drive is not mounted (the consent dialog needs the account owner), so everything lives on the VM's local
  disk and would be lost if the runtime were recycled.
- Code reaches the VM as an xz-compressed, base64 tar bundle typed into notebook cells in pieces of at most
  9 KB; each piece is checked by SHA-256 in the browser before it runs, the whole payload by MD5 on the VM
  (`dist/make_bundle.py`, `dist/split_cell.py`). Text files are normalised to LF line endings when bundled
  (a CRLF copy of a shell script had failed with `$'\r': command not found`).

### Data acquisition
- **Tsinghua Dogs** from the official URLs (cg.cs.tsinghua.edu.cn, cloud.tsinghua.edu.cn): the 2.66 GB
  `low-resolution.zip` needed resumed downloads (local log `research_log/download_tsinghua*.log`, finished
  16:52:55 UTC); the VM fetched its own copy the same way.
- **Stanford Dogs**: the official host (vision.stanford.edu) did not answer (curl exit 7 / 28) and image-net.org
  was also unreachable. `scripts/download_data.sh` falls back to Internet Archive snapshots **of the official
  URLs** (`web.archive.org/web/<timestamp>id_/<official URL>`; images and annotation 20250724000413, lists
  20241221112710) and accepts a file only if its SHA-256 equals the checksum recorded for the official release
  in TensorFlow Datasets. All three matched: `images.tar` 793,579,520 B (a443fc2f…ec58), `annotation.tar`
  21,852,160 B (662ed36f…0fd), `lists.tar` 481,280 B (34b47cac…d6c). Each file keeps a `.sha256` note naming its
  source. The bytes are therefore the official release; only the transport differs.

### Data preparation and within-dataset audit (`scripts/prepare_data.py`, logs `data_*.log` on the VM)
- Every image decoded and hashed (SHA-1, pHash, dHash). Train images that duplicate a test image exactly or
  nearly (pHash Hamming ≤ 4) are excluded from training; a stratified 10 % of the official training images
  (seed 0) becomes the validation split used for all model selection.
- Tsinghua Dogs (17:46 UTC): train 58,627 / val 6,515 / test 5,200 (the official validation split, 40 per breed,
  used only as the final test) / excluded 86. Undecodable 0; exact-duplicate groups 114; pHash near pairs 700, of
  which 87 train–test; near pairs with different labels 113.
- Stanford Dogs (18:08 UTC): train 10,728 / val 1,199 / test 8,580 (official) / excluded 73. Undecodable 0;
  exact-duplicate groups 89; near pairs 156, of which 74 train–test; near pairs with different labels 48.
- Decision: label conflicts are recorded with examples in the audit report, not relabelled (no ground truth to
  arbitrate), and reported as a data-quality finding.

### Features (`scripts/extract_all.sh`, `scripts/extract_features.py`)
- Backbone: DINOv2 ViT-B/14 with registers (timm `vit_base_patch14_reg4_dinov2.lvd142m`), 224 px, frozen.
  g = [CLS ; mean patch] (1536-d); the 256 patch tokens projected on a 256-component PCA fitted on the tokens of
  4,000 **training** images only: 84.2 % of the token variance kept on Tsinghua, 83.3 % on Stanford (fitted on
  Stanford's own training images). The main study uses the first 128 components (nested PCA).
- Views cached: orig (train / val / test), hflip (train); test only: blur, noise, jpeg, lowres, occlude
  (automatic corruptions) and head_occlude, bg_gray, bg_blur, crop_body, crop_head (**annotation-assisted**:
  they use the dataset's boxes, so their results are robustness probes, never automatic inference).
- Tsinghua extraction finished 18:21:59 UTC, peak GPU memory 1.20 GB, about 90–140 images/s (limited by the two
  CPU cores decoding JPEGs). Cache size 24 GB.

### Protocol frozen before any training
- Heads: linear (B0), attentive probe (B1), class attention over all classes (B2), and the re-ranker with the
  queries `class`, `center`, `random` (controls) and `rival`, `rival_soft`, `pairwise` (proposed), K = 5, all
  re-rankers at exactly the parameters of B2 (302,211 on Tsinghua).
- Selection: for each head, the learning rate in {3e-4, 1e-3, 3e-3} with the best validation top-1 on seed 0
  (`run_heads.py --select-lr`, file `experiments/<exp>/lr_selection.json`); then seeds 0, 1, 2 at that rate.
  30 epochs max, early stopping on validation top-1. The test split is not loaded by any training script.
- Hypotheses and decision rules written down in `docs/design.md` §8 before the first run.

### Main experiment launched
- 18:19:40 UTC: `scripts/run_experiment.sh configs/experiments/tsinghua_main.yaml` (lr selection, then 3 seeds).
  Training tokens (orig + hflip, 7.7 GB) kept on the GPU (`max_device_fraction` 0.85); loading took 3.5 min.
- 18:22:10 UTC: the Stanford pipeline started after the Tsinghua extraction ended
  (`scripts/pipeline_stanford.sh`: Stanford caches, Stanford test in the Tsinghua PCA basis for transfer, then
  the cross-dataset overlap audit `scripts/cross_overlap.py`).

### Stanford overlap audit: a crash and its fix
- 18:47:39 UTC: `pipeline_stanford.sh` stopped in its last step. `cross_overlap.py` raised
  `KeyError: id 134 not in cache /content/features/stanford/dinov2_vitb14_reg_224_pca256/orig`. Every Stanford
  cache before it was complete (checked at 18:58: orig 20,507 = train + val + test; hflip 10,728; the eight test
  views 8,580 each; the test split in the Tsinghua PCA basis 8,580).
- Cause: the script asked the cache for a descriptor for every Stanford image. That includes the 73 images the
  split excluded (training images that near-duplicate a test image; id 134 is one), and those have no
  descriptor by design. The Tsinghua side only worked because the script filtered its excluded images there.
- Fix (`scripts/cross_overlap.py`): descriptors are looked up only for the images the cache holds. The others
  keep the pHash signal alone, so they can be candidates but never "confirmed"; the report now records this
  under `with_descriptor`. The clean Stanford test list cannot change, because every Tsinghua train/val image
  and every Stanford test image has a descriptor. `tests/test_cross_overlap.py` now plants one excluded,
  descriptor-less image in each dataset. 26 tests pass locally and on the VM.

### Queues deployed (19:03:19 UTC)
- Code bundle with the evaluation scripts (`final_eval.py`, `ood_eval.py`, `transfer_eval.py`, `explain.py`,
  `efficiency_report.py`, `digest.py`) and two job queues (`scripts/queue.sh`):
  - **A** (`configs/queues/a.txt`), Tsinghua-sized jobs one at a time after `tsinghua_main`: the ablations,
    the open-set split and runs, the other backbones, and the token-width study.
  - **B** (`configs/queues/b.txt`): `stanford_main` (started at once, since its caches were complete), then
    feature extraction for ResNet-50, ConvNeXt-T and CLIP ViT-B/16.
- Learning rates, fixed before these runs:
  - Ablations, open-set and token-width runs borrow the rate selected for the matching head in `tsinghua_main`
    (`lr_selection_from`, with `lr_like` for renamed variants). They therefore add no selection on validation
    data of their own.
  - Each other backbone has its own grid, since one rate does not carry over between feature spaces.
- 19:07:51 UTC: `cross_overlap.py` re-run on its own (`logs/cross_overlap.log`), concurrently with both
  trainings. GPU memory 13.5 / 15.4 GB.

### Cross-dataset overlap: results and a visual check (19:10–19:45 UTC)
- The 19:07:51 run stopped with CUDA out-of-memory: the two trainings held 13.5 of the 15.4 GB. It was re-run
  with `--device cpu` (start 19:10:16, end 19:12:50); the computation is the same, only slower.
- Inputs: 70,428 Tsinghua and 20,580 Stanford images; cached descriptors for 70,342 and 20,507 (the others are the
  split-excluded images and carry pHash only, as designed after the 18:47 fix).
- 155,728 candidate pairs; **19,081 confirmed** (pHash ≤ 8 and cosine ≥ 0.85, or cosine ≥ 0.97), covering
  **18,675 Tsinghua and 18,578 Stanford images (90.3 % of Stanford Dogs)**. This matches the Tsinghua paper's own
  statement that about 18,000 of its images come from Stanford Dogs (Sec. 3.1; `lit_core.md`). 98.8 % of the
  confirmed pairs carry the same breed name. 120 of the 130 Tsinghua breeds have confirmed copies, and within each
  of them the majority Stanford breed accounts for at least 96.2 % of the pairs (median 100 %).
- Threshold sensitivity: over pHash ≤ 4 … 10 and cosine ≥ 0.80 … 0.95 the confirmed Tsinghua images stay between
  18,613 and 18,664. The count does not hinge on the thresholds.
- Confirmed pairs by split (Tsinghua | Stanford): train|train 8,280; train|test 6,003; test|train 1,869;
  test|test 1,305; val|train 946; val|test 678. Stanford test images confirmed in Tsinghua train/val: 6,482 of
  8,580. Stanford test images with no candidate at all in Tsinghua train/val: **973** (the transfer test list).
- Tsinghua side (computed from `cross_pairs.csv` and the manifest): **3,115 of the 5,200 Tsinghua test images
  (59.9 %) have a confirmed Stanford copy**, 3,838 (73.8 %) have at least one candidate. Validation: 1,584 of
  6,515 (24.3 %); training: 13,976 of 58,627 (23.8 %). The test split (the official validation split) is far
  richer in Stanford images than the training data.
- Visual check (random pairs drawn per band, thumbnails rendered in a notebook cell on the VM and inspected on
  screen):
  - confirmed, cosine ≥ 0.97 (18,991 pairs): all 4 inspected pairs are the same photograph;
  - confirmed, pHash ≤ 8 and cosine 0.90–0.97 (88 pairs): both inspected pairs are the same photograph (one with
    a watermark added, one converted to greyscale);
  - rejected, pHash ≤ 8 but cosine < 0.85 (151 pairs, only 13.2 % same breed): both inspected pairs are different
    dogs, i.e. pHash collisions. The rule is right to need both signals;
  - rejected, pHash > 8 and cosine 0.90–0.97 (4,157 pairs, 1,644 Tsinghua images): of 6 inspected, at least 2 are
    the same photograph under a different crop (pHash is not crop-invariant); the others look like different
    photographs of similar dogs.
  - Reading: the confirmed set has high precision but is a **lower bound** on Stanford-origin images. An
    "independent" set must therefore exclude every candidate, not only the confirmed pairs.
- Consequence, recorded before any test evaluation: Stanford Dogs was built from ImageNet and is a subset of the
  ILSVRC training images [krause2016unreasonable, Sec. 5.1]. So **at least 60 % of the Tsinghua test images are
  ImageNet-1k training images**. ResNet-50 and ConvNeXt-T were trained on them with breed labels; DINOv2's
  LVD-142M contains ImageNet-1k and -22k; CLIP's data is not public. Absolute accuracies are optimistic for every
  head. Comparisons between heads share the same frozen features, but a head could still benefit more from
  memorised images than another.
- Added to the evaluation plan (it was already anticipated in `lit_core.md`, "Consequences for our protocol" (1)):
  a **provenance split of the Tsinghua test set**:
  - *Stanford-origin*: a confirmed copy (3,115 images);
  - *independent*: no candidate pair at all (1,362 images);
  - the remaining 723 (candidate only) are counted but not analysed.
  `scripts/subset_eval.py` computes the per-head top-1 and the same paired comparisons on each subset from the saved
  test predictions. Nothing is trained or selected on it. The subsets are fixed by the overlap audit alone, before
  any test prediction exists. The independent subset is not a random sample: it has fewer typical Stanford-like
  photographs and includes the 10 Tsinghua breeds with no confirmed Stanford copy. Its absolute accuracy is not
  comparable to the full test set; only the comparisons between heads on it are.

### Evaluation code locked (19:45–20:11 UTC)
- A smoke-test finding: the `random` control draws its K − 1 random classes afresh at every forward pass, test time
  included, from the global random-number generator. Two evaluations of the same checkpoint therefore gave
  different test top-1 (0.5067 and 0.5029 on the smoke run). Fix in `train_heads.predict`: an optional seed;
  evaluation draws from that seed (the run's own seed) in a forked generator, so it repeats exactly and leaves the
  global generator alone. Training is unchanged. `final_eval.py` also reports the spread of the random control's
  top-1 over 5 draws (seeds s, 1000 + s, …, 4000 + s) as `top1_draw_sd`.
- The freeze is now enforced in code. `scripts/freeze.py` writes `experiments/<exp>/freeze.json`, holding each
  seed run's rate, validation scores and the SHA-256 of its `best.pt`. `final_eval.py`, `ood_eval.py`,
  `transfer_eval.py`, `subset_eval.py` and `explain.py` refuse to run on an experiment without it, on a run it
  does not list, or on a checkpoint whose hash has changed.
- `scripts/final_all.sh` fixes the whole locked evaluation of each experiment: views, comparison pairs and order.
  `scripts/subset_eval.py` handles the provenance split.
- Delivered to the VM as one bundle of 14 files (5 pieces typed into cells, each checked by SHA-256; payload MD5
  9c663e63…). 32 tests pass locally and on the VM. No running script was replaced.

---

## Model selection freeze

*(To be completed from `experiments/*/lr_selection.json` and the seed runs, before `scripts/final_eval.py` is run on
any test split. Nothing below this line may be edited after the first test evaluation.)*

### Analysis rules fixed before any test evaluation (20:18 UTC)
No test prediction exists for any experiment at this point. These choices make the design.md §8 rules mechanical
(implemented in `scripts/make_tables.py`, `best_by_val` and `hypotheses`):
1. **H1.** The "best proposed mode" and "best control" are the tags with the highest mean validation top-1 over
   their three seed runs on **Tsinghua**; ties go to the lower mean validation NLL. Stanford repeats **the same
   pair** ("same sign on Stanford"); its own validation picks are reported beside it, not used for the verdict.
   H2, H5 and H7 use the same proposed head.
2. **H3.** "Within the seed spread" means |random − rival| ≤ the larger of the two test top-1 standard deviations
   over seeds.
3. **The random control** is evaluated with fixed draws (seed = the run's seed); `top1_draw_sd` is reported.
4. **Hard pairs (H2)**: the 20 most-confused breed pairs of `linear` seed 0, from its *validation* predictions.
5. **Provenance subsets**: fixed by the overlap audit above (Stanford-origin 3,115, independent 1,362).
6. **Comparisons**: exactly the pairs in `scripts/final_all.sh`; each judged per seed (paired bootstrap, 10,000
   resamples; exact McNemar) and by the §8 rule.
7. Every experiment's configuration, learning-rate grid and seeds were fixed before its training, as recorded
   above. Nothing is re-selected after a test evaluation.

### stanford_main — frozen 2026-11-08 20:13:39 UTC
`python -I scripts/freeze.py --experiment stanford_main`: 27 runs, 9 heads, every checkpoint hashed into
`experiments/stanford_main/freeze.json`. Validation split: 1,199 images.

| Head | Rate | Val top-1, seeds 0 / 1 / 2 | Mean |
|---|---|---|---|
| linear | 3e-4 | 0.8774 / 0.8749 / 0.8732 | 0.8752 |
| attnprobe | 3e-4 | 0.8590 / 0.8624 / 0.8649 | 0.8621 |
| classattn_all | 3e-4 | 0.8757 / 0.8757 / 0.8757 | 0.8757 |
| rr_class | 3e-4 | 0.8749 / 0.8757 / 0.8766 | 0.8757 |
| rr_center | 3e-4 | 0.8766 / 0.8732 / 0.8782 | 0.8760 |
| rr_random | 3e-4 | 0.8766 / 0.8749 / 0.8791 | 0.8768 |
| rr_rival | 3e-4 | 0.8757 / 0.8732 / 0.8766 | 0.8752 |
| rr_rival_soft | 3e-4 | 0.8749 / 0.8724 / 0.8757 | 0.8743 |
| rr_pairwise | 3e-4 | 0.8757 / 0.8732 / 0.8766 | 0.8752 |

What the validation numbers already say (no test data involved):
- **Every head chose 3e-4, the lowest rate in the grid.** The best rate may lie below the grid. The grid was fixed
  before training and is not widened after the fact; this is reported as a limitation.
- Apart from the attentive probe, all heads lie within 0.8743–0.8768, a spread of 3 images out of 1,199. On
  Stanford validation the re-rankers show no gain over the linear head.
- `rr_rival` and `rr_pairwise` have identical validation top-1 on all three seeds. This is not one model trained
  twice: their validation NLL differs (0.61814 vs 0.61811 on seed 0) and their logits differ by up to 0.041
  (seed 0). But their predicted class agrees on all 1,199 validation images. On Tsinghua the two modes differ
  (seed 0 at 3e-4: 0.8686 vs 0.8682). Checked at 20:16 UTC from `val_preds.npz`.
- Stanford's own validation picks (reported, not used for H1): proposed `rr_pairwise` (ties `rr_rival` at
  0.8752; mean validation NLL 0.61648 vs 0.61651); control `rr_random`.

### stanford_main — locked test evaluation (20:18:56–20:50:58 UTC)
`bash scripts/final_all.sh stanford_main` (log `$WORK/logs/final_stanford_main.log`), started after the freeze above
was recorded: the clean test view with the fixed comparison pairs (20:18:56), the five automatic corruptions
(20:31:17), the three annotation-assisted views (20:46:19); finished 20:50:58. No head, rate, epoch or rule was
changed after it started.

### tsinghua_main — frozen 2026-11-08 21:03:42 UTC
Training finished 20:59:37. `python -I scripts/freeze.py --experiment tsinghua_main`: 27 runs, 9 heads, every
checkpoint hashed into `experiments/tsinghua_main/freeze.json`. Validation split: 6,515 images. Recorded here before
`scripts/final_all.sh tsinghua_main` was started; no Tsinghua test prediction exists yet.

| Head | Rate | Val top-1, seeds 0 / 1 / 2 | Mean |
|---|---|---|---|
| linear | 3e-4 | 0.8678 / 0.8686 / 0.8692 | 0.8686 |
| attnprobe | 3e-4 | 0.8682 / 0.8672 / 0.8689 | 0.8681 |
| classattn_all | 3e-4 | 0.8675 / 0.8685 / 0.8683 | 0.8681 |
| rr_class | 3e-4 | 0.8698 / 0.8723 / 0.8711 | 0.8711 |
| rr_center | 3e-4 | 0.8706 / 0.8703 / 0.8697 | 0.8702 |
| rr_random | 3e-4 | 0.8692 / 0.8691 / 0.8695 | 0.8693 |
| rr_rival | 3e-4 | 0.8686 / 0.8694 / 0.8700 | 0.8693 |
| rr_rival_soft | 3e-4 | 0.8692 / 0.8720 / 0.8718 | 0.8710 |
| rr_pairwise | 3e-4 | 0.8682 / 0.8705 / 0.8697 | 0.8694 |

What the validation numbers already say (no test data involved):
- **The H1 pair, by rule 1 above: `rr_rival_soft` (best proposed, mean 0.8710) against `rr_class` (best control,
  0.8711).** Stanford repeats this pair. The two are level on validation: 2.6130 vs 2.6132 summed over the seeds,
  under one image of 6,515 on average.
- **Every head chose 3e-4 again**, the lowest rate in the grid; the same limitation as on Stanford.
- All six re-rankers lie above the three baselines on validation (0.8693–0.8711 against 0.8681–0.8686), by
  0.07–0.30 pp, or 5–20 images. Among the re-rankers, the proposed modes are not above the controls: the class
  query is the best re-ranker. The seed standard deviations (about 0.0005–0.0016) are as large as the gaps between
  re-rankers.

### Analysis code after the Tsinghua freeze (21:10–21:20 UTC)
The Tsinghua validation pick for H1 (`rr_rival_soft`, above) is not the head the presentation code had been written
around (`rr_rival`). Three changes make the outputs follow rule 1. None changes a number, the definition of a
comparison or the decision rule, and none uses a test result:
- `scripts/make_tables.py`: the comparison table now lists every pair `final_all.sh` evaluates on the clean view
  (the 20 pairs of the forest plot plus the two against B1; before: 14 pairs, centred on `rr_rival`); the forest
  plot takes the same list; the H1 pair carries a dagger; the provenance table puts the H1 pair and the other rule-1
  comparisons of the picked head first.
- `scripts/digest.py`: per-class accuracies are also kept for `rr_rival_soft`, so the per-breed figure can show the
  picked head.
- `scripts/explain.py`: `rr_rival_soft` added to the heads whose attention is measured against the boxes (the
  figure's heads, `rr_rival` and `rr_class`, are unchanged). Patched on the VM at 21:12 UTC, before
  `final_all.sh tsinghua_main` reached that step.
The secondary experiments (ablation, open set, backbones, token width) were configured before the pick, with
`rr_rival` as the representative proposed head, and stay as they are; their tables name the head they use.
While checking the launch of the Tsinghua evaluation (21:05 UTC), its first log lines (clean-view test top-1 of a
few seed runs) were on screen; the changes above follow from the validation pick alone.
Figures: the x label of `fig_main` ran 27 px past the figure edge and was shortened. A text-collision check of the
five figures drawn so far from real data (`/content/figcheck.py` on the VM) finds no overlaps.

### Erratum (21:25 UTC)
The overlap audit above says "at least 60 % of the Tsinghua test images are ImageNet-1k training images". The count
behind it is 3,115 of 5,200 = 59.9 %, so it should read "at least 59.9 %" (a lower bound, as the confirmed set is a
lower bound on the Stanford-origin images). The paper and the README use 59.9 %.

### Queue change (21:24 UTC)
Queue B finished at 20:41 (all three backbone extractions done). To use the idle GPU share, the three backbone
experiments (`bb_resnet50`, `bb_convnext_tiny`, `bb_clip_vitb16`) were moved, with their commands unchanged, from
queue A to a new queue C (`configs/queues/c.txt`), which runs beside queue A; `bb_resnet50` started at 21:24:27.
Queue A keeps ablate → open-set split → open set → token width. No configuration, seed or grid changed.

### Stanford Dogs ∩ ILSVRC-2012 train, by file id (21:28 UTC)
The count planned in `lit_core.md` (Part 4, item 4). Stanford Dogs keeps the ImageNet file names
(`n02085620_5927.jpg`), so an image is an ILSVRC-2012 training image when its id is in the official training list.
List: `train.txt` of BVLC Caffe's `caffe_ilsvrc12.tar.gz` (dl.caffe.berkeleyvision.org; 17,858,008 bytes;
1,281,167 lines, the ILSVRC-2012 training count; SHA-256 of train.txt begins 55a0a92c27626550). Script:
`scripts/ilsvrc_overlap.py`; output `results/audit/ilsvrc_overlap.json`.
- All 120 Stanford Dogs classes are ILSVRC-1k classes.
- **Every Stanford Dogs image is an ILSVRC-2012 training image**: 12,000 of 12,000 official training images and
  **8,580 of 8,580 official test images**.
- Through the confirmed copies, **all 3,115 Stanford-origin Tsinghua test images** (59.9 % of the 5,200) are
  ILSVRC-2012 training images (also 13,976 Tsinghua training and 1,584 validation images).
Consequence (as anticipated above, now exact): ResNet-50 (torchvision `IMAGENET1K_V2`) and ConvNeXt-T
(`IMAGENET1K_V1`) were trained on every Stanford Dogs test image with its breed label; DINOv2's LVD-142M includes
the ImageNet-1k training images (used without labels). Absolute accuracies on Stanford
Dogs and on the Stanford-origin part of Tsinghua Dogs are optimistic for every frozen backbone here; the independent
subset (1,362 images) is the cleaner check. Comparisons between heads share the same features.

### Automatic freeze and evaluation of the secondary experiments (declared 21:42 UTC, before any of them finished)
The runtime is unattended for long stretches, and a free-tier session can end without warning. So from here on a
watcher, `scripts/auto_final.sh`, finishes each secondary experiment (ablation, open set, the three backbones,
token width) as soon as its queue job is marked done:
1. freeze: `scripts/freeze.py --experiment <exp>`, printout kept in `logs/freeze_<exp>.log`;
2. locked test evaluation: `scripts/final_all.sh <exp>`, whose per-experiment plan was fixed before these
   experiments started (rr_rival as the representative proposed head; see "Analysis rules fixed before any test
   evaluation").
No choice is made between the end of training and the freeze, so doing it by machine changes nothing but the
timing. **One deviation from the rule in `final_all.sh`'s header** ("after the freeze is recorded in the research
log"): the freeze summaries are copied into this log *after* the evaluations, from `freeze.json`, whose
`frozen_at_utc` and the first timestamp of `logs/final_<exp>.log` show the order. A failed job or a refused freeze
is logged and skipped, never retried. Once every experiment is done, the watcher runs the efficiency report (the GPU
is quiet then), the digest, the tables and the figures.

### Repository-only changes for a fresh reproduction (21:49 UTC)
`configs/queues/c.txt` gained a first job, `wait_b` (wait until queue B's `extract_clip_vitb16` is marked), so
that all three queues can be started at once. The VM's copy, which queue C is reading, was left unchanged: in the
recorded run queue B had finished (20:41) before queue C started (21:24), so `wait_b` would have passed at once.
The notebooks (`notebooks/make_notebooks.py`) now start queue C too, and notebook 3 has an optional cell that
starts `scripts/auto_final.sh`.

### The GPU runtime ends: Colab usage limit (about 22:14 UTC)
The free-tier T4 runtime was disconnected at about 22:14 UTC (the terminal's last clock tick). Colab then
refused a new GPU runtime: "You cannot currently connect to a GPU due to usage limits in Colab." No paid
compute was bought. The VM's local disk is gone; the Drive mirror (`colab_sync.sh push`, every 10 minutes)
holds `/content/work` as of its last push at 22:09:58 UTC.
- **Finished before the cutoff:** `tsinghua_resnet50`: trained 21:24:27–21:55:21, frozen automatically at
  2026-11-08 21:56:00 UTC (12 runs, 4 heads; every head chose 3e-4 on the seed-0 grid), locked test evaluation
  21:56:02–21:57:39 (`logs/final_tsinghua_resnet50.log`). Freeze summary, validation top-1 per seed:
  linear 0.8459 / 0.8464 / 0.8482 (mean 0.8468); rr_class 0.8467 / 0.8464 / 0.8476 (0.8469);
  rr_pairwise 0.8474 / 0.8483 / 0.8474 (0.8477); rr_rival 0.8480 / 0.8487 / 0.8476 (0.8481).
- **Cut off mid-way:** `tsinghua_ablate` (started 21:00:22; the runs finished by 22:09:58 are on Drive) and
  `tsinghua_convnext_tiny` (started 21:55:21). Neither was frozen; no test split of either was read.
- **Not started:** the open-set split and `tsinghua_openset`, `tsinghua_clip_vitb16`, `tsinghua_tok32`,
  `tsinghua_tok64`, and the efficiency report.
The feature caches were on the lost local disk; a new GPU session rebuilds them from the archives on Drive
(notebook 1, last section), and the queues then skip every finished run. Until the GPU quota returns, a CPU-only
runtime (connected about 22:20 UTC) is used for what needs no GPU: restoring `/content/work` from Drive, the
digest, the tables, the figures and the paper. It trains nothing and evaluates nothing.

### Erratum to the 21:28 entry: what LVD-142M contains (22:33 UTC)
The 21:28 entry says DINOv2's LVD-142M "includes the ImageNet-1k training images (used without labels)", and the
19:45 overlap entry says it "contains ImageNet-1k and -22k". DINOv2's Table 15 lists ImageNet-22k "as is"
(14,197,086 images) and uses the ImageNet-1k training split as a *retrieval seed*. The accurate statement: LVD-142M
contains ImageNet-22k, which holds the ImageNet images Stanford Dogs was drawn from, so the Stanford Dogs images
were in DINOv2's pre-training data without labels. This is an inference from the published composition, not a
count. The paper and the README use this wording. The conclusion (absolute accuracies are optimistic for every
backbone here) is unchanged.

### Paper draft and small repository fixes (22:33 UTC)
The paper agent finished `paper/main.tex` (written before reading any result; every result sentence marked
`\pending`). Fixes made after its report:
- `paper/merge_bib.py` now leaves the `note` fields (how each entry was verified) out of `refs.bib`, so the
  reference list does not print them. They stay in `refs_core.bib` and `refs_related.bib`. The three arXiv-only
  entries keep a note with their arXiv id. No other field changed (checked by diff).
- `scripts/auto_final.sh` now draws the figures with `scripts/make_figures.py` itself. On the lost VM it would have
  used `/content/figcheck.py`, an overlap checker outside the repository; it never reached that step. It also
  passes the five curve runs the README lists, not three. Neither change affects a result.

### Drive mounted on the CPU runtime; Colab GPU still refused (23:11–23:25 UTC)
The owner granted Drive access at about 23:08 UTC. The first mount call had already given up (its wait ran out
before the click). Run again at 23:11, it mounted without a new prompt. `/content/cpu_reports.sh`, a session
helper (not in the repository), started at 23:13:31. It pulls `/content/work` from the mirror without checkpoints,
then runs the digest (five curve runs), `make_tables.py` and `make_figures.py`, exactly the commands of
`scripts/auto_final.sh`. It keeps the GPU-runtime digest of 21:39 as `results/digest_gpu_2139`.
A GPU check at 23:25 UTC, from a scratch notebook so the CPU runtime kept running, got "You cannot currently
connect to a GPU due to usage limits in Colab" again. The paid accelerators are greyed out: this account has no
paid Colab plan.
Using a second Google account for Colab was considered and not done. Colab's FAQ lists "using multiple accounts
to work around access or resource usage restrictions" among the disallowed activities.

### Moving the remaining GPU work to Kaggle: plan and resume rule (declared 23:35 UTC, before any drift result)
The owner offered their Kaggle account. It is phone-verified and has its weekly 30 GPU hours unused. Kaggle is a
separate service with its own free quota, so this works around no Colab limit. The work left needs a GPU:
- `tsinghua_ablate`: 39 runs; part finished on Colab.
- `tsinghua_convnext_tiny`: 12 grid and 8 seed runs; part finished.
- the open-set split and `tsinghua_openset`: 12 runs.
- `tsinghua_tok32` and `tsinghua_tok64`: 9 runs each.
- `tsinghua_clip_vitb16`: 20 runs.
- the efficiency report.
Without a GPU, the finished experiments' digest, tables, figures and paper continue on the CPU runtime.
- **Same data and splits.** The manifests, i.e. the splits, come from the Drive mirror unchanged
  (`scripts/make_resume_bundle.sh`). Tsinghua Dogs is downloaded from the official site on Kaggle, and each archive
  must match the SHA-256 of the copy the Colab runs used, or nothing runs (`scripts/kaggle_run.sh`).
- **Same features as far as possible.** The feature caches were on the lost disk and must be re-extracted. The
  saved token PCA bases are reused, so only the backbone forward pass is recomputed, on other hardware and software.
- **Drift check, validation only.** `scripts/feature_drift_check.py` applies every frozen `tsinghua_main` head,
  and every finished run of the two interrupted experiments, to the rebuilt validation features. It compares
  the result with the validation predictions saved at training time. No test split is read.
- **Resume rule, fixed now.** An interrupted experiment keeps its finished runs only if no deterministic finished
  run's validation top-1 moved by more than 0.001 (0.1 pp, about 7 of 6,515 images) on the rebuilt cache. Otherwise
  its directory moves to `experiments/<exp>_before_rebuild` and the whole experiment is trained again, so that every
  run of one experiment uses the same features (`scripts/resume_policy.py`). The main experiment's drift is reported,
  not acted on: its selection is frozen and its test evaluation is done.
- **Same protocol.** The same configurations, queue commands (`configs/queues/kaggle/`, the a/c jobs unchanged
  plus the extraction and drift jobs) and freeze-then-evaluate watcher (`scripts/auto_final.sh`, with the six
  remaining experiments; its reports reduced to the efficiency report). The digest, tables and figures are then
  made from the Drive mirror, where every result is.
- **Credentials.** No API key or token goes onto either machine. Files move between Drive and Kaggle through a
  temporary folder on the owner's PC, with the owner's permission.

## 2026-11-09

### Resume bundle and reports moved to the owner's PC through the Colab page (23:53–00:01 UTC)
The CPU runtime packed three files from the Drive mirror:
- `resume.tar` (478,208,000 bytes; `scripts/make_resume_bundle.sh`): manifests, the saved token PCA bases, the frozen
  `tsinghua_main` heads with their validation predictions, the finished runs of `tsinghua_ablate` (22) and
  `tsinghua_convnext_tiny` (6), the queue logs;
- `reports.tgz` (3,247,713 bytes): the digest, tables, figures and logs of the CPU reports (23:11–23:25 entry);
- `expected_sha256.txt` (255 bytes): the SHA-256 of the three Tsinghua archives on the Drive mirror
  (`raw/tsinghua_dogs/`), i.e. the copies the Colab runs used.
Method: a small HTTP server on the runtime (port 8123, serving only `/content/relay`) was reached through Colab's
own port proxy from the Colab page in the owner's signed-in browser. A script in that page fetched each 32 MiB part
and PUT it to a relay on the PC that listens only on 127.0.0.1 and answers only the Colab and Kaggle origins (a
session helper, not in the repository). Transfer 23:54:55–00:00:59. Each file's SHA-256 matches on both ends:
`resume.tar` `dd4ea6c32360e93b…` (15 parts), `reports.tgz` `29b2d19720a4264c…`, `expected_sha256.txt`
`3ca848ac7780cd3c…`. At 01:16, two of the three archive hashes were recomputed on the Drive copies and match
(`TrainValSplit.zip`, `low-annotations.zip`; `low-resolution.zip` is 2,664,983,287 bytes and was not re-hashed).

### CPU reports: outcome (started 23:13:31 UTC)
Digest, tables and figures all ended with exit code 0. Digest: 13 files, each listed with its SHA-256 in
`results/digest/MANIFEST.txt`. Tables: 17 written; missing, because their experiments have no test result,
`tab_ablate_other`, `tab_tokens`, `tab_openset`, `tab_efficiency`. Figures: 6 written; `fig_ksweep` missing. Copied
into the repository at 00:46 UTC (`results/digest/`, `paper/generated/`, `paper/figures/`); all 13 digest files match
the manifest on the PC.

### A public Drive link for Kaggle, tried and withdrawn (about 00:13–00:44 UTC)
A Kaggle notebook cannot reach the owner's PC or Colab's proxy, and a Kaggle upload needs a file picked in the
owner's browser. To let a notebook download the three files without any credential, they were placed in a Drive
folder and shared as "anyone with the link", for a bootstrap script that would fetch them by id and check each
SHA-256. The assistant's permission check allowed two of the four share requests and refused the other two. The
route was dropped: the two files already shared were overwritten with 0 bytes, and the folder was moved to the Drive
trash (both confirmed at about 00:44 UTC). The files held only this project's code, manifests, PCA bases,
checkpoints and logs, and no credential. The Kaggle route is now a private dataset that the owner uploads.

### Kaggle bundle and run script (00:45 UTC)
`kaggle_upload/` on the PC holds `repo_tgz.bin` (the repository, LF-normalised, 124,397 bytes,
`a4a91adbd4122668…`), `resume_tar.bin` (= `resume.tar`) and `expected_sha256.txt`. The `.bin` names stop Kaggle
from unpacking the archives on upload. Changes to `scripts/kaggle_run.sh` since the 23:35 plan:
- the work folder, the images and the feature caches live on the scratch disk (`/kaggle/tmp`); only the packed
  results and the logs are kept in `/kaggle/working`;
- a deadline guard (default 630 minutes) stops the queues and the watcher before Kaggle's 12-hour limit and packs
  what exists; an unfinished run keeps its `last.pt`;
- the ablation and the ConvNeXt-T jobs start only after `scripts/resume_policy.py` has written its decision for
  them (the resume rule of 23:35);
- the `.bin` names are accepted.
Unit tests: 36 passed. Waiting for the owner to upload the three files as a private Kaggle dataset.

### Erratum: when Drive was mounted on the GPU runtime (01:17 UTC)
The compute note near the top of this log says Drive is not mounted. That was true when it was written. Drive was
mounted on the GPU runtime at about 21:13 UTC, after the owner approved it (the session's terminal clock when the
mount returned). From then on, `/content/work` was pushed to Drive every 10 minutes; the last push was at 22:09:58
(22:14 entry). The paper's limitation paragraph said Drive was never mounted; it is corrected.

### Paper filled from the digest (00:46–01:17 UTC)
- The abstract, introduction, results and conclusion are written from `paper/generated/numbers.json`, the 17 tables
  and `results/digest`. The rule's outcomes on the corrupted views and on the provenance subsets were recomputed
  from `comparisons.csv` and `subset_comparisons.csv` with the rule of Eq. (12); they agree with the tables.
- Verdicts as computed: H1 refuted; H2 supported on Tsinghua (refuted on Stanford); H3 supported by its criterion,
  though the paired rule makes no claim for rival over random; H4 no difference; H5 supported (re-ranker ahead on
  Tsinghua, no difference on Stanford); H6 not run; H7 exploratory.
- New in the repository: `requirements-colab-full.txt`, the runtime's full `pip freeze` (723 lines). It was recovered
  from the Drive mirror's copy (written 21:28:56) through the Drive connector's text view and un-escaped. The result
  has the Drive file's size, 14,256 bytes; SHA-256 `e9b05fe86464f4d1…`. Also new: `configs/expected_sha256_tsinghua.txt`.
- Token variance kept by the first 32/64/128/256 PCA components, from the saved bases: Tsinghua
  46.4/58.8/71.7/84.2 %, Stanford 43.3/56.3/70.0/83.3 %.
- Compiled on the CPU runtime (pdflatex, bibtex, two more passes): 49 pages, no errors, no undefined references.
- Still pending in the paper: H6 and the K sweep, the other ablations, the token widths, the open set, CLIP ViT-B/16
  and ConvNeXt-T, and the efficiency report.
- Noticed while writing, no action taken: on Stanford Dogs pairwise and rival make almost the same decisions
  (per-seed differences +0.00/+0.01/+0.01 pp). The contrastive heads' attention is close to uniform (relative entropy
  0.96–0.98), unlike the class and centred queries (0.73–0.79). Every head in all three finished experiments chose
  the lowest learning rate of the grid.

### Final compile and README status (01:20–01:26 UTC)

- `README.md`, *Results and status*: the state is now 01:20 UTC on 9 November, with one row per experiment (done,
  interrupted by the GPU limit, not started), the ResNet-50 results (B0 85.59 ± 0.11, class query 85.76 ± 0.01,
  rival 85.85 ± 0.05, pairwise 85.79 ± 0.03 %; rival − class +0.09 pp, no claim by the rule), and links to the
  digest, the full `pip freeze` and the archive checksums. Two wordings corrected against the digest: H1's interval
  clears zero in one seed, not none; Stanford rival-soft − B0 is −0.09 pp, not −0.08. Added: by the paired rule,
  rival-soft beats random rivals (+0.69 pp, all three intervals above zero); rival does not (+0.31 pp, one).
- `paper/main.tex` (only the header comment and the `\date` line changed since the 01:17 build) sent to the CPU
  runtime; SHA-256 `93d3aa72…` on both ends. The 31 files in `paper/generated/` and `paper/figures/` on the VM are
  byte-identical to the repository's. Built at 01:23:45 UTC (pdflatex, bibtex, two more passes): 49 pages, no
  errors, no undefined references or citations. Left as they are: one 12.5 pt overfull line in a table's URL cell,
  and three BibTeX warnings (`volume` and `number` both set in three AAAI entries; only the volume is printed).
- `paper/main.pdf` replaced by this build: 1,447,956 bytes, SHA-256 `2d18c73c980c1ca9…`, the same on both ends.
  Page 1 looked at: title, the new date line and the abstract render.

### Colab GPU checked again (01:29–01:30 UTC): still refused

A new, empty notebook (`Untitled71.ipynb`, so the CPU runtime holding the LaTeX build stayed untouched) was set to
T4 GPU and connected. Colab answered "Cannot connect to GPU backend", "due to usage limits in Colab", about 3 h 15 min
after the GPU runtime ended. The dialog was closed without taking the CPU fallback, and the tab was closed. The
remaining GPU work stays planned for Kaggle (23:35 entry); it waits for the three files in `kaggle_upload/` to be
uploaded to Kaggle, which has to be done by hand: the Kaggle page cannot read from the PC's bridge (requests to
localhost are blocked as `ERR_BLOCKED_BY_CLIENT`), and the browser tools have no file picker.

### Kaggle: bundle checked, setup stage passed on CPU, GPU run started (01:40–02:05 UTC)

- The three files were uploaded by the user, from their own browser, as the private Kaggle dataset
  `hafedefheij/dogbreed-phd-resume` (version 1, 478.33 MB). No API key or token was used on either machine.
- Notebook `hafedefheij/notebookad87bc5fb1` (private), dataset attached, Internet on. Its first cell checks the
  SHA-256 of `repo_tgz.bin`, `resume_tar.bin` and `expected_sha256.txt` against the values written down before the
  upload (`a4a91adb…`, `dd4ea6c3…`, `3ca848ac…`), unpacks the repository, and runs `scripts/kaggle_run.sh` with
  `STAGE=all` when `nvidia-smi` finds a GPU and `STAGE=setup` (stop after the data check) when it does not.
- First run on CPU (no accelerator), as a dry run: all three bundle hashes OK; the Tsinghua Dogs archives,
  downloaded again from the official site (about 4 min), match the three SHA-256 values recorded on Colab; the
  resume state unpacked (`registry.csv`, `tsinghua_main`, `tsinghua_ablate`, `tsinghua_convnext_tiny`); SETUP DONE at
  01:53:52 UTC. The unit tests then passed on Kaggle (every test a dot, no F or E, `[100%]`).
- The CPU session was stopped before the accelerator was changed, so no GPU time went on an idle session. Accelerator
  set to GPU T4 x2 (Kaggle showed 30 of 30 weekly GPU hours left). Version "GPU resume run 2026-11-09" saved with
  *Save & Run All (Commit)* at about 02:04 UTC; Kaggle reports it running on GPU T4 x2. It runs the two queues of
  `configs/queues/kaggle/` (GPU 0: DINOv2 re-extraction, drift checks, the rest of the ablations, open set, 32 and 64
  tokens; GPU 1: ConvNeXt-T and CLIP ViT-B/16), with the 630-minute deadline guard, then packs
  `kaggle_results_small.tar`, `kaggle_results_full.tar` and `kaggle_results.sha256` in `/kaggle/working`.
- No result from this run exists yet. The interrupted experiments keep their finished runs only if the drift check
  passes (resume rule of 23:35).

### Kaggle GPU run: finished (02:04–03:54 UTC), read from its logs; results not yet merged

The version "GPU resume run 2026-11-09" ended "Successfully ran in 6618.8s" (1 h 50 min of the 30 weekly GPU hours).
Read on Kaggle's output page, from `logs/` (nothing downloaded yet); the archives' own numbers are checked when they
are merged.
- **Environment.** Two Tesla T4 (15,360 MiB, driver 580.178.04), 31 GB RAM, 967 GB free on the scratch disk.
  Python 3.13.16, torch 2.11.0+cu130 (CUDA 13.0, cuDNN 92700), timm 1.0.31, numpy 2.1.3, scikit-learn 1.6.1,
  scipy 1.16.3, pandas 2.3.3. The Colab runs used torch 2.14.1+cu130 (cuDNN 92800), so the forward pass is not
  bit-identical, which is what the drift check is for.
- **Data.** The three Tsinghua archives were downloaded again (02:05–02:09) and each matched the SHA-256 of the
  Colab copy. The queues started at 02:09:22.
- **Drift check, validation only (`logs/drift_*.log`).** On the rebuilt DINOv2 cache, the 24 deterministic frozen
  heads of `tsinghua_main` move their validation top-1 by at most 0.0003 (at most 3 of 6,515 predictions change).
  The random-rival control changes 59–82 predictions per seed; it is left out of the rule, as declared. The 22
  finished `tsinghua_ablate` runs move by at most 0.0003 and the 6 finished `tsinghua_convnext_tiny` runs by 0.0000.
  The resume rule (limit 0.001) therefore **kept** the finished runs of both experiments, which then trained only
  their missing runs.
- **Frozen, then tested once (`logs/freeze_*.log`, `logs/final_*.log`).** Every head chose the learning rate 3e-4
  again.

  | experiment | frozen (UTC) | runs | test |
  |---|---|---|---|
  | `tsinghua_convnext_tiny` | 02:40:25 | 12 seed runs, 4 heads | 02:40:59 |
  | `tsinghua_ablate` | 03:06:03 | 39 runs, 13 heads | 03:07:13 |
  | `tsinghua_clip_vitb16` | 03:08:16 | 12, 4 heads | 03:08:48 |
  | `tsinghua_openset` | 03:23:51 | 12, 4 heads | 03:24:10 |
  | `tsinghua_tok32` | 03:38:12 | 9, 3 heads | 03:38:40 |
  | `tsinghua_tok64` | 03:52:42 | 9, 3 heads | 03:53:03 |

- **Test top-1, mean over 3 seeds, as the logs print them (the paired comparisons are in the archives, not in the
  logs).**
  - ConvNeXt-T: pairwise 87.88, rival 87.86, class 87.65, B0 86.95 %.
  - CLIP ViT-B/16: rival 79.83, pairwise 79.62, class 79.51, B0 78.24 %.
  - Ablation: rival at K = 2/3/10: 86.85/86.91/86.52; class at K = 2/3/10: 86.83/86.73/86.42; pairwise at
    K = 3/10: 87.06/86.67; rival without true-class insertion 86.81, without token dropout 86.76, with the
    candidate-only loss 86.60; pairwise with the pair temperature at 0 (hard minimum) and 3 (softer): 86.65/86.76 %.
  - Token width: at 32 dimensions rival 86.58, class 86.49, B1 86.07 %; at 64 dimensions rival 86.63, class 86.56,
    B1 85.90 %.
  - Open set (100 known breeds, 4,000 known and 1,200 unknown test images): top-1 on the known breeds 89.8–90.0 %
    for every head; AUROC of the maximum softmax probability 0.806–0.809, of the maximum logit 0.811–0.818.
- **Efficiency (`logs/efficiency.log`, Tesla T4).** Backbone 85.7 M parameters, 47.1 GFLOPs, 8.4 ms per image at
  batch 1. Heads at batch 1: B0 0.09 ms, rival 0.76 ms, rival-soft 0.86 ms, pairwise 1.03 ms, B2 0.53 ms; at
  batch 256: B2 14.7 ms against 6.4–6.6 ms for the class, rival and rival-soft re-rankers, which therefore take
  under half of B2's time. Full single-image pipeline 7.8–9.1 ms for every head.
- **Packed.** `kaggle_results_small.tar` 6,113,280 bytes, SHA-256 `bc84641d536f0d19…`; `kaggle_results_full.tar`
  1,205,340,160 bytes, SHA-256 `82fbbd4f67a5bce4…`. The unit tests passed again after the run.
- **Not yet done:** bringing the archives to the Drive mirror (`scripts/ingest_kaggle.sh`, written and tested on
  dummy folders), then the digest, tables, figures and the paper. Nothing above is final until the digest is made.

### Kaggle results merged on the Colab runtime, reports made, paper filled (05:40–06:20 UTC)

- **Archives.** The owner downloaded `kaggle_results_small.tar` from the notebook's output in their own browser
  (05:40 UTC). It matches the session's SHA-256 (`bc84641d…`) on the PC and, after the transfer, on the Colab
  runtime. `kaggle_results_full.tar` (1.2 GB) was not found on the PC; it is still kept as the notebook's output.
- **Code check.** The code the Kaggle session ran (`kaggle_upload/repo_tgz.bin`, `a4a91adb…`) differs from the Colab
  runtime's repository only in `scripts/kaggle_run.sh` and `configs/queues/kaggle/{a,c}.txt`. Those three files and
  `scripts/ingest_kaggle.sh` were copied to the runtime first. No file that trains, evaluates or reports differs.
- **Merge, local only.** `LOCAL_ONLY=1 bash scripts/ingest_kaggle.sh /content/kaggle_in` ran 05:53:31–05:54:06 UTC:
  - hash OK; the registry has 89 new rows and keeps every old row;
  - the six experiment folders and their result folders were replaced in `/content/work`, with the old copies in
    `/content/work/before_kaggle/20261109_055331/`;
  - digest rc 0 (15 files), tables rc 0 (21, none missing), figures rc 0 (7, none missing).

  The Drive mirror was read, not written. Replacing its folders with the small archive's copies would leave the
  Colab-trained ablation and ConvNeXt-T runs without checkpoints until the full archive arrives.
- **Back on the PC.**
  - The digest, tables, figures and `configs/openset/tsinghua_unknown30_s0.json` came back as one archive
    (`88e26e8f…`). All 15 digest files match `MANIFEST.txt`. `results/drift/` (the drift checks and resume
    decisions) and `requirements-kaggle-full.txt` (938 lines) were added.
  - The repository's previous digest, tables, figures, paper and README are kept in
    `archive/before_kaggle_merge_20261109/`.
  - Every figure image except the new `fig_ksweep` is byte-identical to the previous one. The tables `tab_ablate_k`,
    `tab_backbones`, `tab_hypotheses` and `tab_lr` changed; `tab_ablate_other`, `tab_tokens`, `tab_openset` and
    `tab_efficiency` are new.
- **The rule on the new pairs (`comparisons.csv`).**
  - Ablation: rival − class at K = 2, 3, 10 is +0.03, +0.18, +0.10; pairwise − rival at K = 3, 10 is +0.15, +0.15.
    All no claim.
  - CLIP ViT-B/16: class > B0 (+1.27) and rival > B0 (+1.59). No claim for rival − class (+0.32), pairwise − class
    (+0.11) or pairwise − rival (−0.21).
  - ConvNeXt-T: class > B0 (+0.71) and rival > B0 (+0.91). No claim for rival − class (+0.21), pairwise − class
    (+0.23) or pairwise − rival (+0.03).
  - Token width: at 32 dimensions, class > B1 (+0.42) and rival > B1 (+0.51); at 64, class > B1 (+0.66) and
    rival > B1 (+0.74). Rival − class is +0.08 at both, no claim.
- **Hypotheses.**
  - H6 is supported by its pre-registered criterion. Rival's top-1 does not fall strictly from K = 2: 86.85, 86.91,
    86.74, 86.52 %, while recall rises from 94.19 to 99.37 %.
  - The paper calls this support weak: every step lies within the seed spread, the rule has no cross-K pair, and the
    class query does fall strictly from K = 2 (86.83, 86.73, 86.71, 86.42 %).
  - H4 at K = 3 and 10: no claim.
  - H5 now has its measured cost. Per image, rival-soft needs 35.3 MFLOPs against B2's 68.0. A batch of 256 takes
    6.5 ms against 14.7 ms. At batch 1, B2 is faster (0.53 against 0.86 ms).
  - The other verdicts are unchanged.
- **Other numbers.** ConvNeXt-T gives the best heads of the study (pairwise 87.89 %). The paper reads this with the
  pre-training overlap: ConvNeXt-T was trained with labels on ILSVRC-2012, which holds 3,115 of the 5,200 Tsinghua
  test images. On the open set, every head separates the 30 held-out breeds about equally well (MSP AUROC
  80.63–80.85 %, FPR95 65.6–67.4 %); no paired test was registered for it.
- **Paper.** Every pending passage is filled from the digest: abstract, introduction, the selection section (new
  paragraph "Resumption on Kaggle"), H4–H6, ablations, open set, other backbones, efficiency, limitations, conclusion
  and appendices (environment, commands, timeline with 9 November, status). No `\pending` remains. It compiles on the
  Colab runtime to 55 pages with no undefined reference. Two warnings remain, and these edits caused neither. The
  attention figure is 31 pt too tall for its page; its PDF has the same size and page box as before, only the
  creation date differs. A URL in the data table runs 12.5 pt into the margin, as recorded at 01:23.
- **Erratum to the previous entry.** In its freeze table, the "test" column gives the time each test evaluation
  *ended*, not when it began. Per `logs/auto_final_kaggle.log`, each began within a second of its freeze (02:40:26,
  03:06:04, 03:08:16, 03:23:52, 03:38:13, 03:52:43) and ended at the times listed.
- **Still open.**
  - Bring the full archive (checkpoints and prediction arrays) to the Drive mirror, then run the ingest without
    `LOCAL_ONLY` to update the mirror.
  - Until then, the Drive mirror holds the pre-Kaggle digest.

### Full Kaggle archive: the uploaded copy comes from a second run; not merged (08:55–09:20 UTC)

- **New Colab runtime.** The runtime used at 05:53 had been recycled; Colab listed no assignment. A CPU runtime
  (2 cores, 12 GB RAM, 108 GB disk) was connected at 08:55 UTC. The repository and the PC's reports came over
  through the page relay as one archive (160 files, `35cf883d…`), and so did the small archive (`bc84641d…`, matched).
  TeX Live was installed. Drive mounted at the second attempt; the first stopped with "Account context authorization
  failed" before anything was read or written.
- **Upload.** The user downloaded `kaggle_results_full.tar` again from the notebook's output page and uploaded it to
  `My Drive/DogBreed-PhD-Research/kaggle_in/` (complete at 09:08 UTC, 1,205,340,160 bytes, the expected size).
- **Hash: no match.** SHA-256 `44670d50…` for the copy in the PC's Downloads, the copy on Drive and the copy on the
  runtime alike, against `82fbbd4f…` written by the 02:04 run. The transfers are therefore not the cause. The small
  archive still matches.
- **Why.** This archive was packed by a second execution of the notebook:
  - its `logs/kaggle_run.log` starts at 05:43:20 UTC (`stage=all`, two T4) and ends "packing (exit code 0)" at
    07:30:59;
  - the six experiments were trained again from the same inputs, then frozen and test-evaluated (freezes from
    06:19:00 to 07:29:24, against 02:40:25 to 03:52:42 in the first run);
  - its newest files are dated 07:30 and the small archive's 03:54; the files both carry over from the Colab resume
    archive keep their 8 November dates.
  
  Nothing in this session started it; a second saved version of the notebook would run exactly this. It used about
  1 h 48 min more of the weekly Kaggle GPU hours. The 02:04 run is the saved version "GPU resume run 2026-11-09"; its
  output page should still hold the archive with the recorded hash.
- **The second run against the first** (every file except checkpoints and prediction arrays; the 02:04 run's small
  archive against the 05:43 run's archive):
  - 237 files are byte-identical;
  - in the 213 that differ, only timing fields and time stamps differ: `train_sec` (87 JSON files), `epoch_sec` in the
    training curves (2,066 values), the latency columns of the efficiency report, `frozen_at_utc` (6), log lines and
    registry times;
  - so do the drift-check draws of the three random-rival control runs, which draw rivals at random by design
    (declared at 23:35);
  - matched by experiment and run, the registry has the same 219 runs. Every other column is identical, validation
    top-1, top-5, NLL, candidate recall, best epoch and learning rate included;
  - every test metric and every confusion matrix is identical, and so is every freeze summary apart from its time
    lines;
  - the only file names in one run and not the other are the time-stamped config snapshots (one per experiment).
  
  The 05:43 run is therefore an exact repeat of the 02:04 run in every number. It also evaluated the test split again,
  after freezing on validation exactly as the first run did. It selected the same heads and nothing was re-selected,
  so the reported results remain the 02:04 run's.
- **Not merged.** The merge was to go ahead only if the archive matched the recorded hash, and it does not. The Drive
  mirror is unchanged. The second run's archive stays in `kaggle_in/`, holding 117 checkpoints (124 MB) and 198
  prediction arrays.

### The second run's checkpoints and predictions checked against the first run's records (09:20–09:33 UTC)

- **The check.** A session helper (`verify_rerun.py`) read the second run's full archive and the first run's small
  archive on the Colab runtime; it trains nothing and runs no network:
  - each `best.pt` was hashed and set against the SHA-256 the first run's freezes recorded (`sha256_best_pt`);
  - each `val_preds.npz` was turned back into the training code's validation metrics and set against the first
    run's `summary.json`;
  - each `test_orig.npz`, with the same run's `val_preds.npz` for the temperature, was turned back into the
    evaluation code's full report (`metrics.full_report`) and set against the first run's `test_metrics_orig.json`.
- **Checkpoints.** All 93 frozen checkpoints (every seed run in the six experiments' freezes) have the SHA-256 the
  first run recorded, so they are byte-identical to the first run's. Of the other 24, all learning-rate grid runs
  (seed 0) of `tsinghua_clip_vitb16` and `tsinghua_convnext_tiny`:
  - the 8 at the chosen rate 3e-4 are the same files as the frozen seed-0 runs, which are copied from them by
    design;
  - the 16 at 1e-3 and 3e-3 have no recorded hash.
- **Predictions.** All 117 validation arrays give the first run's validation top-1, top-5, global top-1 and
  candidate recall exactly, and its NLL to within 1.2e-7: float32 rounding, recomputed on a CPU where it was
  recorded on a GPU. All 81 test arrays give the first run's test report exactly in every count-based metric, the
  per-breed accuracies, ECE, Brier and AURC, and its NLL and temperature to within 2.2e-16.
- **Conclusion.** As far as the first run recorded them, the checkpoints and prediction arrays of the uploaded archive
  are the first run's. The archive's own hash differs only because the second run's result files carry their own
  timings. Nothing has been merged: the condition for merging was a matching archive hash.

### Drive mirror completed: first-run results, second-run checkpoints and predictions (09:45–10:02 UTC)

- **Decision (owner, about 09:45 UTC).** Merge the second run's checkpoints and prediction arrays, checked against the
  first run's records, rather than fetch the first run's archive again. The reported numbers stay the first run's.
- **Step 1: the first run's result files.** `scripts/ingest_kaggle.sh` (Drive written) ran from 09:50:50 to
  09:51:38 UTC, stamp `20261109_095050`. Its input folder held only the first run's `kaggle_results.sha256` and small
  archive:
  - the small archive matched its hash, and the registry gained 89 rows;
  - on Drive and in `/content/work`, the six experiments' folders and results replaced the old ones. The old ones are
    in `before_kaggle/20261109_095050/`, nothing deleted; on Drive these were `tsinghua_ablate` (22 Colab runs),
    `tsinghua_convnext_tiny` (6) and the digest;
  - the archive and its hash file were copied to `kaggle_runs/20261109_095050/` and checked again there;
  - digest, tables and figures all ended with rc 0. Against the repository's, the 16 digest files and 23 tables are
    byte-identical, and the 14 figures are identical apart from the PDF dates of 7. No reported number changes.
- **Step 2: the checkpoints and prediction arrays.** `scripts/merge_rerun_files.py` is new, with 6 unit tests that
  pass locally and on the runtime. It ran check-only at 09:52:32–09:53:42, then for real at 09:54:21–09:56:24:
  - 315 files were checked before any was written:
    - 93 checkpoints against the SHA-256 in `freeze.json`;
    - 8 grid checkpoints against the seed-0 copy's hash;
    - 16 grid checkpoints at the other rates through their validation predictions;
    - 117 validation arrays against `summary.json` (NLL within 1.2e-7);
    - 81 test arrays against `test_metrics_orig.json` (within 2.2e-16).
  - 630 copies were written (work folder and mirror), with nothing already present and nothing overwritten;
  - the manifest `merge_rerun_20261109_095417.tsv` (path, bytes, SHA-256, record, deviation) is on Drive in
    `kaggle_runs/20261109_095050/` and `work/logs/`, and in the repository in `results/kaggle_merge/` with the logs.
- **Checks afterwards** (a session helper, reading only):
  - all 315 files read back from both places with the manifest's size and hash;
  - every file of the small archive is on the mirror byte for byte (334 experiment files, 60 result files, 61
    logs, the open-set split and its backup, the Kaggle `pip freeze`);
  - every file in `before_kaggle/20261109_095050/` still has a live counterpart, and all 28 Colab checkpoints are
    identical to the live ones. The only differences are the two partial `val_preds.npz` of the runs cut off at
    about 22:14 on 8 November (`tsinghua_ablate/rr_pairwise_k10_s1`, `tsinghua_convnext_tiny/rr_rival_lr0.0003_s0`):
    the old folders hold that file only, and Kaggle trained both runs in full. The old digest differs from the
    new, as it should;
  - the live experiment folders now hold a checkpoint for every run: ablation 39, CLIP 24, ConvNeXt 24, open set 12,
    32 and 64 tokens 9 each (the grid runs included); `stanford_main`, `tsinghua_main` and `tsinghua_resnet50` are
    untouched.
- **On Drive's servers.**
  - `drive.flush_and_unmount()` ran in a terminal at 10:00:12–10:00:15 UTC with exit 0, so no write was still
    pending. Drive is unmounted on the runtime since then.
  - Through the Drive connector, the manifests are there with their sizes, and so are four sampled
    `test_orig.npz`, with the manifest's sizes.
  - A provenance note, `kaggle_runs/20261109_095050/README.txt`, says what each file there is.
- **Kept as they were.** The second run's archive stays in `kaggle_in/` (`44670d50…`). The first run's full archive
  (`82fbbd4f…`) never reached Drive and is not needed now.
- **Tests.** 42 pass locally and on the runtime: 36 before this session and the 6 new ones.
- **Closed.** The open item of the 05:40 entry: the Drive mirror now holds the Kaggle results and digest, and a
  checkpoint and prediction arrays for every run.
- **Documents.** The README, the paper (Appendix A, Tables 31 and 32) and this log were updated. Table 31's new rows
  were kept short, and one older row lost two words ("from then on"), so that the table fits its page. The paper was
  compiled on the runtime at 10:13 UTC: 55 pages and no undefined reference. The same two warnings remain as before:
  the attention figure is 31 pt too tall, and a URL runs 12.5 pt into the margin. `paper/main.pdf` was replaced:
  1,528,805 bytes, SHA-256 `b5606596467a30e9…`, the same on both ends.
