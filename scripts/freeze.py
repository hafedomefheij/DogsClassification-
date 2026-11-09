"""Freeze model selection for an experiment, before any of its test evaluations.

    python -I scripts/freeze.py --experiment tsinghua_main

Writes experiments/<experiment>/freeze.json:
* when (UTC);
* the learning rate of every head (lr_selection.json, or the rate borrowed through lr_selection_from);
* for every finished seed run: its validation metrics, best epoch, and the SHA-256 of its best.pt.
The scripts that read a test split (final_eval.py, ood_eval.py, transfer_eval.py, explain.py, subset_eval.py)
refuse an experiment without this file, and any run whose checkpoint changed after it was written
(dogbreed.evaluation.runs.require_freeze). Refuses to overwrite an existing freeze unless --force (that would be a
protocol event and belongs in the research log).
Prints a compact summary for research_log/research_log.md ("Model selection freeze").
"""
from __future__ import annotations

import argparse
import sys
import time

from dogbreed import paths
from dogbreed.evaluation.runs import FREEZE, run_dirs, sha256_file
from dogbreed.utils import load_json, save_json

VAL_KEYS = ("val_top1", "val_top5", "val_nll", "val_top1_global", "val_cand_recall", "best_epoch", "epochs_run")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--expect-seeds", type=int, default=3, help="refuse if a head has fewer finished seeds")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    exp_dir = paths.EXPERIMENTS / a.experiment
    out = exp_dir / FREEZE
    if out.exists() and not a.force:
        raise SystemExit(f"{out} exists; a second freeze needs --force and an entry in the research log")
    runs = run_dirs(exp_dir)
    if not runs:
        raise SystemExit(f"no finished runs in {exp_dir}")
    rec, per_tag = {}, {}
    for tag, seed, d in runs:
        s = load_json(d / "summary.json")
        rec[d.name] = {"tag": tag, "seed": seed, "lr": s.get("config", {}).get("lr"),
                       **{k: s.get(k) for k in VAL_KEYS}, "sha256_best_pt": sha256_file(d / "best.pt")}
        per_tag.setdefault(tag, []).append(rec[d.name])
    short = {t: len(v) for t, v in per_tag.items() if len(v) < a.expect_seeds}
    if short:
        raise SystemExit(f"heads with fewer than {a.expect_seeds} finished seeds: {short}")
    sel = exp_dir / "lr_selection.json"
    fz = {"experiment": a.experiment, "frozen_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
          "lr_selection": load_json(sel) if sel.exists() else None,
          "lr_by_tag": {t: sorted({r["lr"] for r in v}) for t, v in per_tag.items()},
          "runs": rec}
    save_json(fz, out)
    print(f"FREEZE {a.experiment} {fz['frozen_at_utc']} UTC: {len(rec)} runs, {len(per_tag)} heads")
    for t, v in sorted(per_tag.items()):
        vt = [r["val_top1"] for r in v]
        print(f"  {t:<22} lr {','.join(f'{x:g}' for x in fz['lr_by_tag'][t])}  val top-1 "
              + " / ".join(f"{x:.4f}" for x in vt) + f"  mean {sum(vt) / len(vt):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
