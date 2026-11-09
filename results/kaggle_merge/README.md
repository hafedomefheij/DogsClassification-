# How the Kaggle results reached the Drive mirror (9 November 2026)

The six secondary experiments were finished in a Kaggle session from 02:04 to 03:54 UTC (notebook version
"GPU resume run 2026-11-09"). It packed two archives and wrote their SHA-256 to `kaggle_results.sha256`:
`kaggle_results_small.tar` (`bc84641d…`, every result file except checkpoints and prediction arrays) and
`kaggle_results_full.tar` (`82fbbd4f…`). Every reported number comes from this session.

The notebook later ran a second time, from 05:43 to 07:31 UTC, from the same inputs. It repeated every step and gave
the same selections and the same validation and test metrics; only timings and time stamps differ. Its full archive
(`44670d50…`, 1,205,340,160 bytes) is the one that reached Drive, as `kaggle_in/kaggle_results_full.tar`. Its own hash
is not the first session's, so it was not merged as a whole. The mirror was completed in two steps on a Colab CPU
runtime:

1. `bash scripts/ingest_kaggle.sh` (Drive written) with the first session's small archive and hash file only:
   - the hash matched and the registry gained 89 rows;
   - the six experiments' folders and results replaced the old ones on Drive and in `/content/work`. The old ones
     went to `work/before_kaggle/20261109_095050/`, nothing deleted;
   - the archive and its hash file were copied to `kaggle_runs/20261109_095050/` and checked again there;
   - the digest, tables and figures came out identical to the repository's. The 16 digest files and 23 tables match
     byte for byte; 7 figures differ only in the PDF date stamps;
   - log: [`ingest_20261109_095050.log`](ingest_20261109_095050.log).
2. `python -I scripts/merge_rerun_files.py`: the 315 checkpoints and prediction arrays of the second run's archive,
   each checked against the first session's records before anything was written. The check-only pass, then the
   merge, wrote 315 files into the work folder and 315 onto the mirror, overwriting nothing. Logs:
   [`merge_rerun_check_only.log`](merge_rerun_check_only.log),
   [`merge_rerun_20261109_095417.log`](merge_rerun_20261109_095417.log).

| Files | Checked against | Result |
|---|---|---|
| 93 `best.pt` of the frozen runs | the SHA-256 in the experiment's `freeze.json` | all equal |
| 8 `best.pt` of grid runs at the chosen rate | the SHA-256 of the seed-0 run copied from them | all equal |
| 16 `best.pt` of grid runs at the other rates | no recorded hash; their `val_preds.npz` (next row) | passed |
| 117 `val_preds.npz` | the run's `summary.json`, recomputed from the arrays | counts exact, NLL within 1.2e-7 |
| 81 `test_orig.npz` | the run's `test_metrics_orig.json`, recomputed with the evaluation code | counts and per-breed accuracy exact, other numbers within 2.2e-16 |

[`merge_rerun_20261109_095417.tsv`](merge_rerun_20261109_095417.tsv) lists every file with its size, SHA-256, the
record it was checked against and the largest deviation; the `.json` holds the totals and the archive's hash. Both are
also on Drive in `kaggle_runs/20261109_095050/` and `work/logs/`. Afterwards:
- every merged file was read back from the work folder and from the mirror with its manifest hash;
- every file of the small archive was found on the mirror byte for byte;
- the old folders were compared with the new. All 28 checkpoints trained on Colab (22 ablation, 6 ConvNeXt) are
  identical in both. The only old files that differ are two partial `val_preds.npz` of the runs cut off on Colab at
  about 22:14 on 8 November, which Kaggle trained again in full; they remain in `before_kaggle/`;
- Drive's pending writes were flushed (`drive.flush_and_unmount`, exit 0) and a sample was checked on Drive's servers.
