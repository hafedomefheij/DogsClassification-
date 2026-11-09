# DogBreed-PhD-Research — instructions for any Claude session opened here

A PhD research project: an original, generalisable learning mechanism for fine-grained visual classification
(FGVC), with dog breeds as the main application and birds (CUB-200-2011) and aircraft (FGVC-Aircraft) as the
generalisation domains. GitHub: https://github.com/hafedomefheij/DogsClassification-.git (branch `main`).

## Start here after a restart
1. Read `SESSION_STATE.md` (what was running, where), then `NEXT_ACTIONS.md` (what to do next, in order).
2. `RECOVERY_GUIDE.md` says how to restore checkpoints, resume a Kaggle run and re-verify files.
3. `git status` and `git log --oneline -5`; compare with the commit named in `SESSION_STATE.md`.
4. Do not repeat a finished experiment: check `EXPERIMENT_REGISTRY.md` and `CHECKPOINT_MANIFEST.json` first.

## Two studies
- **Study 1 (finished, frozen):** RCEA, a K = 5 shortlist re-ranker over frozen DINOv2 patch tokens, on Tsinghua and
  Stanford Dogs. Code `src/dogbreed/`, paper `paper/main.pdf`, results `results/digest/`, log
  `research_log/research_log.md`. Do not edit its code, results or paper except to fix a documented error.
- **Study 2 (in progress):** the continuation prompt of 2026-11-09 (the original method, three domains, modern
  baselines). Research documents in `research/`, decisions in `RESEARCH_DECISIONS.md`. New code goes in a new
  package beside `src/dogbreed/`, never over Study 1's files.

## Rules (from the owner; not negotiable)
- Real experiments only. Never invent a number, curve, figure, citation or log. Pending work is labelled pending.
- Test sets are read only after model selection is frozen and recorded; no test information in training or tuning.
- Hypotheses and the analysis plan for Study 2 are committed before any CUB / Aircraft test evaluation.
- No API key or token on any machine (no Kaggle token, no Hugging Face token, no GitHub token in files).
- No paid compute, no billing change. Sign-ins, OAuth and licence/consent clicks are the owner's.
- No CAPTCHAs. Never click "Disconnect and delete runtime" in Colab. Public Drive sharing was refused: do not retry.
- Respect dataset and model licences: no dataset images, feature arrays or model weights in Git.
- Never force-push. Commit verified milestones; end commit messages with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Verify every push with `git ls-remote origin`.
- Do not touch `C:\Users\ECS\Documents\SS` (another project). Use absolute paths: the shell's folder sometimes resets.
- Ordinary technical choices are made without asking; the owner's messages are short and informal.

## Compute (checked 2026-11-09; see RESEARCH_DECISIONS.md D3)
- **Kaggle** (primary; owner's account, phone verified): about 26 of 30 weekly GPU hours left on 9 Nov, T4 x2 or
  P100, 12 h per session, Internet on. Code reaches Kaggle by `git clone` of this public repository at a pinned
  commit; datasets are downloaded inside the notebook from their official URLs and checked by SHA-256; results come
  back as the notebook's output files (downloaded by the owner) and as summaries printed in the log.
- **Colab** (backup; free tier on the owner's Google account): GPU sometimes refused by usage limits. A paid Google
  AI Pro benefit was not visible on that account.
- **This PC**: CPU only (4 cores, 16 GB, torch 2.15 CPU in `.venv`): tests, small checks, documents.

## Layout
`src/` code · `configs/` configurations · `scripts/` run and report scripts · `tests/` pytest · `notebooks/` Colab
and Kaggle notebooks · `paper/` LaTeX and PDF · `research/` literature review, gap, novelty, hypotheses, method ·
`research_log/` dated log · `results/` digests (small files only) · `archive/` earlier copies kept for provenance.
Ignored by Git: `data/`, `experiments/`, `checkpoints/`, `figures/`, `kaggle_upload/`, `for_supervisor/`, `.venv/`.

## Checks before a commit
`.venv/Scripts/python.exe -m pytest -q tests` passes; no file over 5 MB; no secret (see `RECOVERY_GUIDE.md`, "Before
committing").
