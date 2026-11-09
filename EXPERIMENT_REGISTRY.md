# Experiment registry

One row per experiment. Status: **done** (results in the digest), **failed**, **running**, **pending** (designed,
not run). Every number in a paper or report must trace to a *done* row. Per-run details (seed, learning rate,
checkpoint hash, git commit) are in the files named in the last column.

## Study 1 — RCEA on frozen backbones, Tsinghua and Stanford Dogs (finished 2026-11-09)
| Experiment | Backbone | Runs | Status | Where the records are |
|---|---|---|---|---|
| `tsinghua_main` | DINOv2 ViT-B/14-reg, frozen | 54 | done (Colab) | `results/digest/runs.csv`, `test.csv` |
| `stanford_main` | DINOv2 ViT-B/14-reg, frozen | 54 | done (Colab) | same |
| `tsinghua_ablate` | DINOv2 ViT-B/14-reg | 39 | done (22 Colab + Kaggle) | same; `results/kaggle_merge/` |
| `tsinghua_resnet50` | ResNet-50, frozen | 24 | done | same |
| `tsinghua_convnext_tiny` | ConvNeXt-T, frozen | 24 | done (6 Colab + Kaggle) | same |
| `tsinghua_clip_vitb16` | CLIP ViT-B/16, frozen | 24 | done (Kaggle) | same |
| `tsinghua_openset` | DINOv2 ViT-B/14-reg | 12 | done (Kaggle) | same |
| `tsinghua_tok32`, `tsinghua_tok64` | DINOv2, 32 / 64 PCA dims | 9 + 9 | done (Kaggle) | same |
Checkpoints and prediction arrays: on the owner's Drive mirror (`DogBreed-PhD-Research/work/experiments/…`) and in
the Kaggle notebook output of version "GPU resume run 2026-11-09"; per-file SHA-256 in
`results/kaggle_merge/merge_rerun_20261109_095417.tsv` and each experiment's `freeze.json`. See
`CHECKPOINT_MANIFEST.json`.

## Study 2 — original mechanism, three domains (started 2026-11-09)
| ID | What | Status |
|---|---|---|
| S2-LIT | Literature and code search (4 search agents + synthesis), `research/*.md` | running |
| S2-DATA | CUB-200-2011 and FGVC-Aircraft from official URLs: SHA-256, manifests, split ids, integrity audits | pending |
| S2-FEAT | Frozen features for DINOv2-reg, DINOv3, SigLIP 2 (and supervised ResNet-50 / ConvNeXt / ViT) | pending |
| S2-LP | Setting 1: frozen backbone + linear classifier, validation-selected, every backbone × dataset | pending |
| S2-METHOD | Setting 3: matched backbone + the proposed mechanism | pending (method not chosen yet) |
| S2-COMPETE | Setting 4: matched backbone + closest established mechanism(s) | pending |
| S2-FT | Setting 2: fine-tuned backbones (fewer seeds; compute-limited) | pending |
| S2-ABL | Ablations that test the mechanism's explanation | pending |
| S2-GEN | Cross-dataset transfer and distribution-shift tests | pending |
Rules: test metrics of CUB and Aircraft are computed only after `research/PREREGISTRATION.md` is committed and the
run's selection is frozen (RESEARCH_DECISIONS.md D2).
