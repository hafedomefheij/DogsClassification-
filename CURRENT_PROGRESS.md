# Current progress

*Updated 2026-11-09 11:20 UTC.* Completed, failed and pending work are kept apart; nothing here is a result unless
it says **done** and names its record.

## Study 1 — done
RCEA (shortlist re-ranking over frozen DINOv2 patch tokens), Tsinghua and Stanford Dogs, 249 runs, 55-page paper
(`paper/main.pdf`). Main test top-1 (mean ± sd over 3 seeds, `results/digest/test.csv`): Tsinghua B0 86.26 ± 0.21,
rival-soft 87.12 ± 0.15; Stanford all heads about 88.3. H1 refuted; H2, H3, H5 supported on Tsinghua only. Integrity
findings: 19,081 Stanford copy pairs; 3,115 of 5,200 Tsinghua test images are copies; every Stanford image is in
ILSVRC-2012 train.

## Study 2 — stage gates (continuation prompt 2, section 14)
| Gate | Question | State |
|---|---|---|
| 1 Novelty | Is the mechanism meaningfully different from prior work? | open — literature search running |
| 2 Correctness | Does the implementation match the mathematics? | open — nothing implemented yet |
| 3 Experimental value | Measurable benefit over controlled baselines? | open — no experiment run |
| 4 Mechanism validation | Do ablations support the explanation? | open |
| 5 Generalisation | Useful beyond dogs (birds, aircraft)? | open |
| 6 Reproducibility | Can others reproduce it? | open |
| 7 Academic strength | Contribution, evidence, limitations at doctoral level? | open |

## Study 2 — work items
- [x] Existing work inspected; GitHub repository was empty; Study 1 committed and pushed (`347ac3f`).
- [x] Compute checked: Kaggle primary (about 26 GPU h this week), Colab free backup, AI Pro not visible.
- [x] Model access checked: DINOv3 (timm copy) and SigLIP 2 downloadable without tokens.
- [ ] Literature and code search, `research/*.md` (running).
- [ ] Candidate comparison and choice of mechanism.
- [ ] Mathematics, implementation, tests.
- [ ] CUB and Aircraft data, audits; features; baselines; method; ablations; generalisation.
- [ ] Paper / thesis chapter for Study 2.
