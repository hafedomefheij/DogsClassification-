"""Locked final evaluation on the TEST split (run only after model selection is frozen).

    python -I scripts/final_eval.py --experiment tsinghua_main --views orig
    python -I scripts/final_eval.py --experiment tsinghua_main --views blur,jpeg,occlude,bg_gray \
        --tags linear,rr_rival,rr_class
    python -I scripts/final_eval.py --experiment tsinghua_main --compare rr_rival:linear,rr_rival:rr_class

For every finished run (best.pt) and view: test logits -> full metrics, with the temperature
fitted on that run's validation predictions (never on test). Per tag: mean, std and t-based
95 % CI over seeds. Comparisons: per seed, paired bootstrap of the accuracy difference and the
exact McNemar test on the same test images.
Outputs: results/<experiment>/<view>/{runs.csv, summary.csv, summary.json, comparisons.json,
hard_pairs.json} and test_<view>.npz next to each run.
Refuses to run before scripts/freeze.py has frozen the experiment's model selection, or on a run whose
checkpoint changed since (dogbreed.evaluation.runs.require_freeze).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from dogbreed import paths
from dogbreed.datasets.manifest import load_manifest
from dogbreed.evaluation import metrics as M
from dogbreed.evaluation import stats as S
from dogbreed.evaluation.runs import apply_class_subset, latest_config, require_freeze, run_dirs
from dogbreed.training.cached import load_split
from dogbreed.training.train_heads import load_head, predict
from dogbreed.utils import auto_device, get_logger, save_json

log = get_logger()
KEY_METRICS = ["top1", "top5", "macro_precision", "macro_recall", "macro_f1", "mean_per_class_acc", "nll",
               "ece", "nll_ts", "ece_ts", "aurc", "aurc_margin", "risk_at_90cov", "brier"]
EXTRA = ["top1_global", "cand_recall", "flip_rate", "fix_rate", "break_rate", "top1_draw_sd"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--views", default="orig")
    ap.add_argument("--tags", default=None)
    ap.add_argument("--compare", default="", help="A:B pairs, comma separated")
    ap.add_argument("--hard-pairs-ref", default="linear", help="tag whose confusions define the hard pairs")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    exp_dir = paths.EXPERIMENTS / a.experiment
    cfg = latest_config(exp_dir)
    device = auto_device(a.device)
    df, names = apply_class_subset(cfg, load_manifest(cfg.get("manifest") or paths.MANIFESTS / f"{cfg['dataset']}.csv"))
    C = len(names)
    runs = run_dirs(exp_dir, set(a.tags.split(",")) if a.tags else None)
    fz = require_freeze(exp_dir, runs)
    log.info("%d runs to evaluate (selection frozen %s UTC)", len(runs), fz["frozen_at_utc"])
    for view in a.views.split(","):
        vdir = paths.feature_dir(cfg["dataset"], cfg["backbone"], cfg.get("img_size", 224), view,
                                 cfg.get("feature_tag", ""))
        if not (vdir / "meta.json").exists():
            log.warning("no cache for view %s (%s): skipped", view, vdir)
            continue
        test = load_split([vdir], df, "test", device, token_dims=cfg.get("token_dims"))
        out_dir = paths.RESULTS / a.experiment / view
        out_dir.mkdir(parents=True, exist_ok=True)
        rows, correct, preds = [], {}, {}
        for tag, seed, d in runs:
            head, ck = load_head(d / "best.pt", device)
            p = predict(head, test, device, seed=seed)          # fixed draws for the random control
            np.savez_compressed(d / f"test_{view}.npz", ids=p["ids"], y=p["y"],
                                logits=p["logits"].astype(np.float32),
                                **({"logits_global": p["logits_global"].astype(np.float32)} if "logits_global" in p else {}),
                                **({"cand": p["cand"]} if "cand" in p else {}))
            pv = np.load(d / "val_preds.npz")
            t = M.fit_temperature(pv["logits"], pv["y"])
            rep = M.full_report(p["logits"], p["y"], C, t=t)
            if "logits_global" in p:
                rep["top1_global"] = M.topk_acc(p["logits_global"], p["y"], 1)
                rep["top5_global"] = M.topk_acc(p["logits_global"], p["y"], 5)
                rep.update(M.rerank_effect(p["logits"], p["logits_global"], p["y"]))
            if "cand" in p:
                rep["cand_recall"] = float((p["cand"] == p["y"][:, None]).any(1).mean())
            if ck["config"]["head"].get("mode") == "random":
                # its rivals are re-drawn at every pass: report how much top-1 moves over 4 more draws
                draws = [rep["top1"]] + [M.topk_acc(predict(head, test, device, seed=1000 * (j + 1) + seed)["logits"],
                                                    p["y"], 1) for j in range(4)]
                rep["top1_draws"] = draws
                rep["top1_draw_sd"] = float(np.std(draws, ddof=1))
            per_class = rep.pop("per_class_acc")
            save_json({"tag": tag, "seed": seed, "view": view, **rep, "per_class_acc": per_class},
                      d / f"test_metrics_{view}.json")
            rows.append({"tag": tag, "seed": seed, **{k: rep.get(k) for k in KEY_METRICS + EXTRA + ["temperature"]}})
            correct[(tag, seed)] = p["logits"].argmax(1) == p["y"]
            preds[(tag, seed)] = p["logits"].argmax(1)
            log.info("%s %s s%d: top1 %.4f top5 %.4f ece %.4f->%.4f", view, tag, seed, rep["top1"],
                     rep["top5"], rep["ece"], rep["ece_ts"])
        if not rows:
            continue
        rdf = pd.DataFrame(rows)
        rdf.to_csv(out_dir / "runs.csv", index=False)
        summ = {}
        for tag, g in rdf.groupby("tag"):
            summ[tag] = {m: S.mean_std(g[m].dropna().tolist()) for m in KEY_METRICS + EXTRA
                         if g[m].notna().any()}
        save_json(summ, out_dir / "summary.json")
        flat = [{"tag": t, **{f"{m}_mean": v["mean"] for m, v in s.items()},
                 **{f"{m}_std": v["std"] for m, v in s.items()}, "seeds": s["top1"]["n"]} for t, s in summ.items()]
        pd.DataFrame(flat).sort_values("top1_mean", ascending=False).to_csv(out_dir / "summary.csv", index=False)

        # paired comparisons on identical test images
        comps = {}
        for pair in [x for x in a.compare.split(",") if x]:
            A, B = pair.split(":")
            per_seed = []
            for (tag, seed), ca in correct.items():
                if tag == A and (B, seed) in correct:
                    cb = correct[(B, seed)]
                    per_seed.append({"seed": seed, **S.paired_bootstrap(ca, cb, seed=seed), **S.mcnemar(ca, cb)})
            if per_seed:
                comps[pair] = {"per_seed": per_seed,
                               "mean_diff": float(np.mean([r["diff"] for r in per_seed])),
                               "seeds_A_better": int(sum(r["diff"] > 0 for r in per_seed))}
        save_json(comps, out_dir / "comparisons.json")

        # hard pairs: the most confused class pairs of the reference model (seed 0), found on its
        # VALIDATION predictions (choosing them on test would favour any model that differs from the
        # reference exactly where the reference was unlucky on test: regression to the mean)
        ref = next(((t, s, d) for (t, s, d) in runs if t == a.hard_pairs_ref and s == 0), None)
        if ref is not None and (ref[2] / "val_preds.npz").exists():
            y = test.y.numpy()
            rv = np.load(ref[2] / "val_preds.npz")
            cm = M.confusion_matrix(rv["y"], rv["logits"].argmax(1), C)
            pairs = M.top_confused_pairs(cm, 20)
            hp = {"pairs_defined_on": f"validation predictions of {ref[0]}_s0",
                  "pairs": [(names[i], names[j], n) for i, j, n in pairs], "by_run": {}}
            for (tag, seed), pr in preds.items():
                errs = sum(int(((y == i) & (pr == j)).sum() + ((y == j) & (pr == i)).sum()) for i, j, _ in pairs)
                hp["by_run"][f"{tag}_s{seed}"] = {"pair_errors": errs, "pair_class_acc": M.pair_accuracy(y, pr, pairs)}
            agg = defaultdict(list)
            for k, v in hp["by_run"].items():
                agg[k.rsplit("_s", 1)[0]].append(v["pair_errors"])
            hp["pair_errors_mean"] = {k: float(np.mean(v)) for k, v in agg.items()}
            hp["pair_class_acc_mean"] = {}
            for t in sorted({k.rsplit("_s", 1)[0] for k in hp["by_run"]}):
                hp["pair_class_acc_mean"][t] = float(np.mean(
                    [v["pair_class_acc"] for k, v in hp["by_run"].items() if k.rsplit("_s", 1)[0] == t]))
            np.save(out_dir / f"confusion_val_{a.hard_pairs_ref}_s0.npy", cm)
            save_json(hp, out_dir / "hard_pairs.json")
        for (tag, seed), pr in preds.items():           # test confusion matrices, seed 0 only
            if seed == 0:
                np.save(out_dir / f"confusion_test_{tag}_s0.npy", M.confusion_matrix(test.y.numpy(), pr, C))
        log.info("view %s summary:\n%s", view,
                 pd.DataFrame(flat).sort_values("top1_mean", ascending=False)[["tag", "top1_mean", "top1_std", "seeds"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
