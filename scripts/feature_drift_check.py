"""Do rebuilt feature caches change what trained heads predict? (validation split only)

    python -I scripts/feature_drift_check.py --experiment tsinghua_main \
        --out $DOGBREED_RESULTS/drift/tsinghua_main.json

A feature cache lives on a runtime's local disk. When it has to be rebuilt on other hardware or software (the
Colab GPU runtime and its disk were lost on 2026-11-08; the caches were re-extracted elsewhere with the same
manifests and the same saved token PCA basis), the heads already trained on the old cache are applied to the
new cache's VALIDATION features and compared with the validation predictions they wrote at training time
(val_preds.npz, from their best epoch). The test split is never read.

Per run: validation top-1 then and now, the number and share of validation images whose predicted class changed,
and the largest absolute logit difference. The random-rival control draws its rivals at random at every pass, so
its "then" and "now" differ by construction; it is listed but left out of the summary.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from dogbreed import paths
from dogbreed.datasets.manifest import load_manifest
from dogbreed.evaluation.runs import apply_class_subset, latest_config
from dogbreed.training.cached import load_split
from dogbreed.training.train_heads import load_head, predict
from dogbreed.utils import auto_device, get_logger, save_json

log = get_logger()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    exp_dir = paths.EXPERIMENTS / a.experiment
    cfg = latest_config(exp_dir)
    device = auto_device(a.device)
    df, _ = apply_class_subset(cfg, load_manifest(cfg.get("manifest") or paths.MANIFESTS / f"{cfg['dataset']}.csv"))
    vdir = paths.feature_dir(cfg["dataset"], cfg["backbone"], cfg.get("img_size", 224), "orig",
                             cfg.get("feature_tag", ""))
    val = load_split([vdir], df, "val", device, token_dims=cfg.get("token_dims"))
    rows = []
    # every finished run, the learning-rate grid runs included (an interrupted experiment may have only those)
    finished = [d for d in sorted(exp_dir.iterdir())
                if d.is_dir() and (d / "best.pt").exists() and (d / "val_preds.npz").exists()]
    for d in finished:
        tag, seed = d.name.rsplit("_s", 1)[0], int(d.name.rsplit("_s", 1)[1])
        head, ck = load_head(d / "best.pt", device)
        now = predict(head, val, device)
        then = np.load(d / "val_preds.npz")
        if not np.array_equal(np.asarray(now["ids"]), then["ids"]):
            raise SystemExit(f"{d.name}: the validation images differ from the ones it was trained with")
        pn, pt = now["logits"].argmax(1), then["logits"].argmax(1)
        rows.append({"run": d.name, "tag": tag, "seed": seed,
                     "random_control": ck["config"]["head"].get("mode") == "random",
                     "top1_then": float((pt == then["y"]).mean()), "top1_now": float((pn == now["y"]).mean()),
                     "changed_images": int((pn != pt).sum()), "changed_share": float((pn != pt).mean()),
                     "max_abs_logit_diff": float(np.abs(now["logits"].astype(np.float64)
                                                        - then["logits"].astype(np.float64)).max())})
        r = rows[-1]
        log.info("%s: val top-1 %.4f -> %.4f; %d of %d predictions changed", d.name, r["top1_then"],
                 r["top1_now"], r["changed_images"], len(pn))
    det = [r for r in rows if not r["random_control"]]
    if not det:
        raise SystemExit(f"{a.experiment}: no finished deterministic run to compare")
    summary = {"experiment": a.experiment, "split": "val", "n_images": int(len(pn)), "cache": str(vdir),
               "summary_excluding_random": {
                   "runs": len(det),
                   "max_abs_top1_diff": max(abs(r["top1_now"] - r["top1_then"]) for r in det),
                   "mean_top1_diff": float(np.mean([r["top1_now"] - r["top1_then"] for r in det])),
                   "max_changed_share": max(r["changed_share"] for r in det),
                   "mean_changed_share": float(np.mean([r["changed_share"] for r in det]))},
               "runs": rows}
    save_json(summary, a.out)
    log.info("summary: %s", summary["summary_excluding_random"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
