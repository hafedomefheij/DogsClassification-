# Next actions (in order)

*Updated 2026-11-09 11:20 UTC.* Tick an item only when its result is verified and recorded.

1. [ ] Commit and push the recovery files; verify with `git ls-remote origin`.
2. [ ] Build the Study 2 data-and-baseline pipeline (method-independent): official downloads with SHA-256,
   manifests with split ids, integrity audits (exact and near duplicates within and across splits and datasets),
   frozen feature extraction for each backbone, linear probes selected on validation only. Unit tests on CPU.
3. [ ] Kaggle notebook: `git clone` at a pinned commit, run stage 0 (data + audits) on CPU first, then features and
   linear probes on GPU. No test metric for CUB / Aircraft yet (D2).
4. [ ] When the four literature agents finish: check a sample of their citations against the sources, then write
   `research/LITERATURE_REVIEW.md`, `RESEARCH_GAP.md`, `NOVELTY_MATRIX.md`, `CLOSEST_PRIOR_WORK.md`,
   `RESEARCH_QUESTIONS.md`, `HYPOTHESES.md`, `SEARCH_LOG.md`.
5. [ ] Compare candidates A–D (originality, closest work, maths, cost, risk, testability, generalisation, reviewer
   objections); a separate novelty check of the chosen mechanism; record the choice in `RESEARCH_DECISIONS.md`.
6. [ ] `research/METHOD.md`: notation, equations, losses, gradients, stability, shapes, complexity, pseudocode,
   limitations. Implement it; gradient checks, overfit tests, forward/backward tests.
7. [ ] `research/PREREGISTRATION.md` committed before any CUB / Aircraft test evaluation.
8. [ ] Kaggle runs: settings 1–4 on the four datasets, seeds, ablations, transfer and shift tests; checkpoints with
   full state; a verified restore.
9. [ ] Analysis, figures, the Study 2 paper; update the registry, the manifest and these files; commit and push.
