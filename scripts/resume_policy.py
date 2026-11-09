"""Keep, or set aside, the runs an interrupted experiment finished before its feature cache had to be rebuilt.

    python -I scripts/resume_policy.py --experiment tsinghua_ablate \
        --drift $DOGBREED_RESULTS/drift/tsinghua_ablate.json

The rule, fixed in the research log before any drift result existed (2026-11-08, "Moving the remaining GPU work
to Kaggle"): the finished runs are kept, and the experiment resumes with only its missing runs, if no
deterministic finished run's validation top-1 moved by more than --max-top1-diff (0.001 = 0.1 percentage point,
about 7 of 6,515 Tsinghua validation images, below the seed spread of every head in the main experiment) on the
rebuilt cache (scripts/feature_drift_check.py). Otherwise the experiment directory is moved to
experiments/<experiment>_before_rebuild and the experiment is trained again from the start, so that every run of
one experiment uses the same features. An experiment with no finished run has nothing to decide.
The decision is written to <drift file>_policy.json. Never reads the test split.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from dogbreed import paths
from dogbreed.utils import get_logger, load_json, save_json

log = get_logger()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--drift", required=True)
    ap.add_argument("--max-top1-diff", type=float, default=0.001)
    a = ap.parse_args(argv)
    exp_dir = paths.EXPERIMENTS / a.experiment
    drift = Path(a.drift)
    out = drift.with_name(drift.stem + "_policy.json")
    if not drift.exists():
        raise SystemExit(f"{drift} not found: run scripts/feature_drift_check.py first")
    s = load_json(drift)["summary_excluding_random"]
    keep = s["max_abs_top1_diff"] <= a.max_top1_diff
    rec = {"experiment": a.experiment, "rule_max_abs_top1_diff": a.max_top1_diff,
           "observed_max_abs_top1_diff": s["max_abs_top1_diff"], "runs_checked": s["runs"],
           "decision": "keep finished runs, train only the missing ones" if keep else
           "set the finished runs aside and train the whole experiment again"}
    if not keep:
        dest = paths.EXPERIMENTS / f"{a.experiment}_before_rebuild"
        if dest.exists():
            raise SystemExit(f"{dest} already exists; refusing to overwrite it")
        shutil.move(str(exp_dir), str(dest))
        rec["moved_to"] = str(dest)
    save_json(rec, out)
    log.info("%s: %s (max |val top-1 change| %.4f, rule %.4f)", a.experiment, rec["decision"],
             s["max_abs_top1_diff"], a.max_top1_diff)
    return 0


if __name__ == "__main__":
    sys.exit(main())
