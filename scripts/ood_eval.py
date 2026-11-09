"""Open-set evaluation with truly held-out breeds (run after model selection is frozen).

    python -I scripts/ood_eval.py --experiment tsinghua_openset

The heads of the experiment were trained with ``class_subset`` (scripts/make_openset_split.py): no
image of an unknown breed was seen in training or validation. Here they score the TEST images of the
known breeds (in-distribution, ID) and of the unknown breeds (out-of-distribution, OOD).

Scores, larger = more in-distribution: maximum softmax probability (raw, and after the temperature
fitted on the run's validation predictions), maximum logit, energy (log-sum-exp of the logits) and
the top-1/top-2 logit margin; for two-stage heads also the same scores of the stage-1 logits.
Metrics: AUROC and FPR at 95 % TPR; closed-set top-1 on the known breeds.
Outputs: results/<experiment>/ood_<view>.json (per run, and mean/std per tag) and ood_<view>_runs.csv.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from dogbreed import paths
from dogbreed.datasets.manifest import load_manifest
from dogbreed.evaluation import metrics as M
from dogbreed.evaluation import stats as S
from dogbreed.evaluation.runs import latest_config, require_freeze, run_dirs
from dogbreed.training.cached import load_split
from dogbreed.training.train_heads import load_head, predict
from dogbreed.utils import auto_device, get_logger, load_json, save_json

log = get_logger()


def score_set(z: np.ndarray, t: float, prefix: str = "") -> dict:
    s = M.confidence_scores(z)
    s["msp_ts"] = M.softmax(z, t).max(1)
    return {prefix + k: v for k, v in s.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--view", default="orig")
    ap.add_argument("--tags", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    exp_dir = paths.EXPERIMENTS / a.experiment
    cfg = latest_config(exp_dir)
    if not cfg.get("class_subset"):
        raise SystemExit("this experiment has no class_subset: nothing is held out")
    split = load_json(cfg["class_subset"])
    known, unknown = sorted(split["known"]), sorted(split["unknown"])
    device = auto_device(a.device)
    df = load_manifest(cfg.get("manifest") or paths.MANIFESTS / f"{cfg['dataset']}.csv")
    df_id = df[df["label"].isin(known)].copy()
    df_id["label"] = df_id["label"].map({c: i for i, c in enumerate(known)})
    df_ood = df[df["label"].isin(unknown)].copy()
    df_ood["label"] = -1                                       # never used: no head knows them
    vdir = paths.feature_dir(cfg["dataset"], cfg["backbone"], cfg.get("img_size", 224), a.view,
                             cfg.get("feature_tag", ""))
    tid = load_split([vdir], df_id, "test", device, token_dims=cfg.get("token_dims"))
    tood = load_split([vdir], df_ood, "test", device, token_dims=cfg.get("token_dims"))
    log.info("ID test %d images (%d breeds), OOD test %d images (%d breeds)", tid.n, len(known),
             tood.n, len(unknown))
    rows, per_run = [], {}
    runs = run_dirs(exp_dir, set(a.tags.split(",")) if a.tags else None)
    require_freeze(exp_dir, runs)
    for tag, seed, d in runs:
        head, _ = load_head(d / "best.pt", device)
        pid, pood = predict(head, tid, device, seed=seed), predict(head, tood, device, seed=seed)
        pv = np.load(d / "val_preds.npz")
        t = M.fit_temperature(pv["logits"], pv["y"])
        sid, sood = score_set(pid["logits"], t), score_set(pood["logits"], t)
        if "logits_global" in pid:
            t0 = M.fit_temperature(pv["logits_global"], pv["y"]) if "logits_global" in pv.files else 1.0
            sid.update(score_set(pid["logits_global"], t0, "g_"))
            sood.update(score_set(pood["logits_global"], t0, "g_"))
        res = {"top1_known": M.topk_acc(pid["logits"], pid["y"], 1), "temperature": t}
        for k in sid:
            for m, v in M.ood_metrics(sid[k], sood[k]).items():
                res[f"{k}_{m}"] = v
        per_run[f"{tag}_s{seed}"] = res
        rows.append({"tag": tag, "seed": seed, **res})
        log.info("%s s%d: top1(known) %.4f  AUROC msp %.4f energy %.4f maxlogit %.4f", tag, seed,
                 res["top1_known"], res["msp_auroc"], res["energy_auroc"], res["maxlogit_auroc"])
    if not rows:
        raise SystemExit("no finished runs")
    rdf = pd.DataFrame(rows)
    out = paths.RESULTS / a.experiment
    out.mkdir(parents=True, exist_ok=True)
    rdf.to_csv(out / f"ood_{a.view}_runs.csv", index=False)
    summ = defaultdict(dict)
    for tag, g in rdf.groupby("tag"):
        for c in g.columns:
            if c not in ("tag", "seed") and g[c].notna().any():
                summ[tag][c] = S.mean_std(g[c].dropna().tolist())
    save_json({"known": len(known), "unknown": len(unknown), "n_id": tid.n, "n_ood": tood.n,
               "split_file": cfg["class_subset"], "per_run": per_run, "summary": summ},
              out / f"ood_{a.view}.json")
    log.info("summary (AUROC, mean over seeds):\n%s", pd.DataFrame(
        {t: {k: v["mean"] for k, v in s.items() if k.endswith("auroc") or k == "top1_known"}
         for t, s in summ.items()}).T.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
