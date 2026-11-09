# Recovery guide

## After the PC or the Claude session restarts
1. Open a Claude session in `C:\Users\ECS\Documents\DogBreed-PhD-Research` (not in `Documents\SS`).
2. Read `SESSION_STATE.md`: it names the last verified commit, what was running (Kaggle notebook and version) and
   which background agents were working.
3. `git status`; `git log --oneline -5`; `git ls-remote origin` must show the same `main` as `git rev-parse HEAD`
   once everything is pushed. Uncommitted work is never discarded: commit it on a branch first.
4. Continue with the first unchecked item of `NEXT_ACTIONS.md`.

## A Kaggle run that was cut off
- Kaggle keeps a notebook version's output after the session ends (also when it stops at the 12-hour limit, if the
  run packed its outputs first; Study 1's runs use a deadline guard for this).
- Study 2 runs write `last.pt` (latest valid, written atomically: temporary file, flush, rename) and `best.pt`
  (best validation) for every run, with the contents listed in `CHECKPOINT_MANIFEST.json`. A new session attached to
  the old version's output resumes from `last.pt` and skips runs whose `done.json` exists.
- Do not re-run a finished run: check `EXPERIMENT_REGISTRY.md` and the run's `done.json`.

## Restoring a checkpoint (and proving it works)
`python -m <package>.restore_check CHECKPOINT` (Study 2) loads the checkpoint into a fresh model and optimizer,
checks the stored SHA-256 of the configuration and split manifest, re-computes the validation metric and compares it
with the stored one. A restore counts as verified only when this check passes; the result is logged in
`CHECKPOINT_MANIFEST.json`.

## Files that live outside Git
- Datasets: downloaded from the official URLs (`scripts/` of each study); archive SHA-256 in `configs/`.
- Study 1 checkpoints and arrays: Drive mirror and Kaggle outputs (see `CHECKPOINT_MANIFEST.json`).
- Study 2 checkpoints and features: Kaggle notebook outputs (persist per version; attachable as notebook inputs).

## Before committing
- `.venv/Scripts/python.exe -m pytest -q tests` passes.
- `git add --dry-run .` lists nothing from `data/`, `experiments/`, `checkpoints/`, no `*.pt`, `*.npz`, `*.zip`.
- No secret in the staged files (GitHub / Google / Hugging Face tokens, private keys, `kaggle.json`); no personal
  e-mail address.
- Commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; push without `--force`;
  check with `git ls-remote origin`.
