# Research decisions

Each decision: date (UTC), what was decided, why, and what would change it. Newest last. Study 1's decisions are in
`research_log/research_log.md`; this file covers Study 2 (the continuation prompt of 2026-11-09) and the decisions
that link the two.

## D1 — Continue, do not restart (2026-11-09)
Study 1 (RCEA on Tsinghua and Stanford Dogs) is finished and frozen as it is: code, 249 runs, digest, 55-page paper.
Study 2 builds on it in new files. Nothing of Study 1 is overwritten; its results are reused only as reported.
*Why:* the owner's instruction ("do not restart, abandon, overwrite"), and Study 1's results are a finding in their
own right (H1 refuted; rival-soft +0.86 pp on Tsinghua; no effect on Stanford; heavy test-set contamination).

## D2 — Fresh confirmatory test sets; pre-registration (2026-11-09)
The Tsinghua and Stanford test results of Study 1 are known. Any Study 2 method is designed after seeing them, so a
gain measured again on those two test sets would be adaptive analysis (a garden of forking paths).
- CUB-200-2011 and FGVC-Aircraft test sets are the **confirmatory** sets of Study 2. Nobody looks at them before
  the hypotheses, the analysis plan, the configurations and the selection rule are committed to Git
  (`research/HYPOTHESES.md`, `research/PREREGISTRATION.md`), and the selection is frozen per run.
- Model selection uses validation data only: FGVC-Aircraft's official `val` split; for CUB, a fixed stratified
  split of the official training set (seed and image ids recorded).
- Tsinghua and Stanford test results of Study 2 are reported as **secondary**, with this caveat stated.
*Changes if:* another unseen dog test set becomes available (then it joins the confirmatory sets).

## D3 — Compute (2026-11-09)
- Kaggle is the primary GPU (owner's preference, 9 Nov: "kaggle better"); about 26 of 30 weekly GPU hours left.
- Colab free tier is the backup. Google AI Pro was not visible on the Colab account in use; the official pages say AI
  Pro gives Colab compute units only in eligible countries. No other Google account is used (owner's rule).
- No paid compute. The budget shapes the design: frozen-backbone experiments on cached features (cheap, many seeds),
  fine-tuning only where the comparison needs it, with fewer seeds, stated as such.

## D4 — Code and data route to Kaggle without tokens (2026-11-09)
The repository is public on GitHub, so a Kaggle notebook with Internet on can `git clone` it at a pinned commit and
check the commit hash. Datasets are downloaded inside the notebook from the official URLs and checked by SHA-256.
No Kaggle API token, no Hugging Face token. Results return as notebook output files that the owner downloads, plus
summaries printed to the notebook log, which can be read in the browser.

## D5 — Backbones and their access (checked 2026-11-09 through the Hugging Face API)
| Backbone | Weights used | Gated | Licence |
|---|---|---|---|
| DINOv2 ViT-B/14 with registers (Study 1) | `timm/vit_base_patch14_reg4_dinov2.lvd142m` / torch hub | no | Apache-2.0 |
| DINOv3 ViT-B/16 | `timm/vit_base_patch16_dinov3.lvd1689m` | no | DINOv3 License (custom) |
| SigLIP 2 ViT-B/16 | `timm/vit_base_patch16_siglip_224.v2_webli` | no | Apache-2.0 |
| ResNet-50, ConvNeXt, ViT (supervised ImageNet) | timm | no | Apache-2.0 / per model card |
Meta's own DINOv3 repository on Hugging Face (`facebook/dinov3-vitb16-pretrain-lvd1689m`) is gated ("manual") and is
not used. The DINOv3 License allows research use; publications must acknowledge the materials, redistribution must
carry the licence (we redistribute no weights), and military / weapons uses are prohibited.

## D6 — What counts as the contribution (2026-11-09, provisional until the literature review is done)
One mechanism, mathematically specified and the same on every domain, judged by the seven stage gates of the
prompt (novelty, correctness, experimental value, mechanism validation, generalisation, reproducibility, academic
strength). A gate that fails is reported and investigated, not argued away. The candidate comparison and the
choice are in `research/NOVELTY_MATRIX.md` and `research/CLOSEST_PRIOR_WORK.md`.
