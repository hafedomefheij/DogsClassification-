"""Test accuracy on provenance subsets of the Tsinghua Dogs test split, from the saved test predictions.

    python -I scripts/subset_eval.py --experiment tsinghua_main --view orig \
        --compare rr_rival:rr_class,rr_rival:rr_random,rr_rival:linear

About 18,000 Tsinghua images are Stanford Dogs photographs (Tsinghua paper, Sec. 3.1), and Stanford Dogs is a
subset of the ILSVRC training images, so part of the test split was in the backbones' pre-training data. The
subsets are fixed by the cross-dataset audit alone (results/audit/cross_pairs.csv, written by cross_overlap.py),
before any test prediction exists:
  all              the whole test split (equals final_eval.py; a consistency check)
  stanford_origin  test images with a CONFIRMED Stanford Dogs copy
  independent      test images with no candidate pair at all (neither signal close to any Stanford image)
Test images with a candidate but no confirmed pair belong to neither subset (counted as "ambiguous").

Reads only test_<view>.npz, written by final_eval.py next to each run; trains and selects nothing.
Output: results/<experiment>/<view>/subsets.json (top-1 per head, mean over seeds, and the paired comparisons
per seed with the same bootstrap and exact McNemar test as final_eval.py).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from dogbreed import paths
from dogbreed.evaluation import stats as S
from dogbreed.evaluation.runs import require_freeze, run_dirs
from dogbreed.utils import get_logger, save_json

log = get_logger()


def provenance_subsets(pairs: pd.DataFrame, test_ids) -> dict:
    """{subset name: sorted test ids}. ``pairs`` needs the columns t_id and confirmed."""
    ids = np.sort(np.asarray(test_ids, dtype=np.int64))
    confirmed = pairs["confirmed"].astype(str).str.lower().isin(["true", "1"])
    conf = np.unique(pairs.loc[confirmed, "t_id"].astype(np.int64))
    cand = np.unique(pairs["t_id"].astype(np.int64))
    in_conf, in_cand = np.isin(ids, conf), np.isin(ids, cand)
    return {"all": ids, "stanford_origin": ids[in_conf], "independent": ids[~in_cand],
            "ambiguous": ids[in_cand & ~in_conf]}


def load_correct(runs, view):
    """{(tag, seed): bool array} over the test images sorted by id, and those ids."""
    correct, ref = {}, None
    for tag, seed, d in runs:
        z = np.load(d / f"test_{view}.npz")
        order = np.argsort(z["ids"])
        ids = np.asarray(z["ids"])[order]
        ok = (z["logits"].argmax(1) == z["y"])[order]
        if ref is None:
            ref = ids
        elif not np.array_equal(ids, ref):
            raise SystemExit(f"{d}: its test predictions cover different images")
        correct[(tag, seed)] = ok
    return correct, ref


def evaluate(correct, ids, subsets, compare, n_boot=10000):
    out = {"n": {k: int(len(v)) for k, v in subsets.items()}, "top1": {}, "comparisons": {}}
    for name, sid in subsets.items():
        if name == "ambiguous":
            continue
        m = np.isin(ids, sid)
        if not m.any():
            continue
        per = {}
        for (tag, seed), ok in sorted(correct.items()):
            per.setdefault(tag, []).append(float(ok[m].mean()))
        out["top1"][name] = {t: S.mean_std(v) for t, v in per.items()}
        comps = {}
        for pair in compare:
            A, B = pair.split(":")
            rows = []
            for (tag, seed), ca in sorted(correct.items()):
                if tag == A and (B, seed) in correct:
                    cb = correct[(B, seed)]
                    rows.append({"seed": seed, **S.paired_bootstrap(ca[m], cb[m], n_boot=n_boot, seed=seed),
                                 **S.mcnemar(ca[m], cb[m])})
            if rows:
                comps[pair] = {"per_seed": rows, "mean_diff": float(np.mean([r["diff"] for r in rows]))}
        out["comparisons"][name] = comps
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--view", default="orig")
    ap.add_argument("--pairs", default=str(paths.RESULTS / "audit" / "cross_pairs.csv"))
    ap.add_argument("--compare", default="", help="A:B pairs, comma separated")
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args(argv)

    runs = [(t, s, d) for t, s, d in run_dirs(paths.EXPERIMENTS / a.experiment)
            if (d / f"test_{a.view}.npz").exists()]
    if not runs:
        raise SystemExit(f"no test_{a.view}.npz in {a.experiment}: run final_eval.py first")
    require_freeze(paths.EXPERIMENTS / a.experiment, runs)
    pairs = pd.read_csv(a.pairs, usecols=["t_id", "confirmed"])
    correct, ids = load_correct(runs, a.view)
    subsets = provenance_subsets(pairs, ids)
    res = evaluate(correct, ids, subsets, [x for x in a.compare.split(",") if x], a.n_boot)
    res.update({"experiment": a.experiment, "view": a.view, "pairs_file": a.pairs,
                "definition": {"stanford_origin": "confirmed Stanford Dogs copy (cross_overlap.py rule)",
                               "independent": "no candidate pair at all",
                               "ambiguous": "candidate pair(s) but none confirmed; not analysed"}})
    out = paths.RESULTS / a.experiment / a.view / "subsets.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json(res, out)
    for name, tops in res["top1"].items():
        log.info("%s (%d images): %s", name, res["n"][name],
                 ", ".join(f"{t} {v['mean']:.4f}" for t, v in sorted(tops.items(), key=lambda x: -x[1]["mean"])))
    log.info("-> %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
