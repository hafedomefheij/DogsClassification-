"""Tsinghua -> Stanford transfer: heads trained on Tsinghua Dogs label the de-duplicated Stanford Dogs
test images, with no Stanford training at all (run after model selection is frozen).

    python -I scripts/transfer_eval.py --experiment tsinghua_main

Breeds: the empirical correspondence written by scripts/cross_overlap.py (for each Tsinghua breed,
the majority Stanford breed among its confirmed duplicate pairs). A Stanford breed is evaluated only
if exactly one Tsinghua breed maps to it with purity >= --min-purity and >= --min-pairs pairs.
Images: Stanford test images that have no candidate duplicate in Tsinghua train or val
(results/audit/stanford_clean_test.json). Tokens: the Stanford test cache in the Tsinghua PCA basis
(feature tag <feature_tag>_tsinghua), so the heads see the token coordinates they were trained on.
The prediction is the full Tsinghua decision over all its breeds; it counts as correct when it is the
mapped breed. A second number restricts the decision to the mapped breeds.
Outputs: results/<experiment>/transfer_stanford.json and transfer_stanford_runs.csv.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from dogbreed import paths
from dogbreed.datasets.manifest import class_names, load_manifest
from dogbreed.evaluation import metrics as M
from dogbreed.evaluation import stats as S
from dogbreed.evaluation.runs import latest_config, require_freeze, run_dirs
from dogbreed.training.cached import load_split
from dogbreed.training.train_heads import load_head, predict
from dogbreed.utils import auto_device, get_logger, load_json, save_json

log = get_logger()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="tsinghua_main")
    ap.add_argument("--min-purity", type=float, default=0.9)
    ap.add_argument("--min-pairs", type=int, default=5)
    ap.add_argument("--tags", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    exp_dir = paths.EXPERIMENTS / a.experiment
    cfg = latest_config(exp_dir)
    if cfg["dataset"] != "tsinghua":
        raise SystemExit("transfer is defined from Tsinghua-trained heads")
    audit = paths.RESULTS / "audit"
    corr = load_json(audit / "cross_tsinghua_stanford.json")["breed_correspondence"]
    clean = set(load_json(audit / "stanford_clean_test.json")["ids"])
    t_df = load_manifest(paths.MANIFESTS / "tsinghua.csv")
    t_names = class_names(t_df)
    t_label = {n: i for i, n in enumerate(t_names)}
    inv = defaultdict(list)
    for tname, v in corr.items():
        if v["purity"] >= a.min_purity and v["pairs"] >= a.min_pairs:
            inv[v["stanford"]].append(tname)
    mapping = {s: ts[0] for s, ts in inv.items() if len(ts) == 1}
    log.info("Stanford breeds with a unique Tsinghua breed: %d (of %d mapped)", len(mapping), len(inv))
    s_df = load_manifest(paths.MANIFESTS / "stanford.csv")
    ev = s_df[(s_df["split"] == "test") & s_df["id"].isin(clean) & s_df["class_name"].isin(mapping)].copy()
    ev["label"] = ev["class_name"].map(lambda s: t_label[mapping[s]])
    vdir = paths.feature_dir("stanford", cfg["backbone"], cfg.get("img_size", 224), "orig",
                             f"{cfg.get('feature_tag', '')}_tsinghua")
    device = auto_device(a.device)
    test = load_split([vdir], ev, "test", device, token_dims=cfg.get("token_dims"))
    mapped_cls = np.array(sorted({t_label[t] for t in mapping.values()}))
    log.info("evaluating %d clean Stanford test images of %d breeds", test.n, len(mapping))
    rows = []
    runs = run_dirs(exp_dir, set(a.tags.split(",")) if a.tags else None)
    require_freeze(exp_dir, runs)
    for tag, seed, d in runs:
        head, _ = load_head(d / "best.pt", device)
        p = predict(head, test, device, seed=seed)
        z, y = p["logits"], p["y"]
        zr = np.full_like(z, -np.inf)
        zr[:, mapped_cls] = z[:, mapped_cls]
        r = {"tag": tag, "seed": seed, "top1": M.topk_acc(z, y, 1), "top5": M.topk_acc(z, y, 5),
             "top1_mapped_only": M.topk_acc(zr, y, 1)}
        if "logits_global" in p:
            r["top1_global"] = M.topk_acc(p["logits_global"], y, 1)
        rows.append(r)
        log.info("%s s%d: top1 %.4f (mapped-only %.4f)", tag, seed, r["top1"], r["top1_mapped_only"])
    rdf = pd.DataFrame(rows)
    out = paths.RESULTS / a.experiment
    out.mkdir(parents=True, exist_ok=True)
    rdf.to_csv(out / "transfer_stanford_runs.csv", index=False)
    summ = {t: {c: S.mean_std(g[c].dropna().tolist()) for c in g.columns
                if c not in ("tag", "seed") and g[c].notna().any()} for t, g in rdf.groupby("tag")}
    save_json({"n_images": test.n, "n_breeds": len(mapping), "min_purity": a.min_purity,
               "min_pairs": a.min_pairs, "mapping": mapping, "summary": summ},
              out / "transfer_stanford.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
