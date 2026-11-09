"""Where the evidence heads look, and whether that is on the dog (run after selection is frozen).

    python -I scripts/explain.py --experiment tsinghua_main \
        --images /content/data/raw/tsinghua_dogs/low-resolution.zip --tags rr_rival,rr_class,attnprobe

For each head (seed 0) and every image of --split:
* its attention over the token grid, averaged over attention heads (pairwise mode: also over the
  candidate's rival pairs). Re-rank heads: the map of the final top-1 candidate. Attentive probe:
  the mean of its queries. All-class attention: the map of the predicted class.
* box statistics, using the dataset's head and body boxes for EVALUATION only: the share of the
  attention mass inside each box, and the box's area share (what a uniform map would give);
  lift = mass / area, so lift > 1 means the map prefers the box.
* a figure on systematically chosen images (no hand-picking): for the first --fig-tag, the first
  --n-show images by id of each kind -- re-ranking fixed the stage-1 error / re-ranking broke a
  correct stage-1 answer / both right -- with the maps of the two strongest final candidates of
  every head in --fig-tags.
Outputs: results/<experiment>/explain/{box_stats.json, shown_maps.json (4-bit quantised maps of the
shown images, for re-plotting elsewhere), explain_<split>.png}
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import torch

from dogbreed import paths
from dogbreed.datasets.corruptions import _box
from dogbreed.datasets.manifest import load_manifest
from dogbreed.datasets.sources import ImageSource
from dogbreed.datasets.transforms import box_to_input_frame, box_token_coverage, default_resize
from dogbreed.evaluation.runs import apply_class_subset, latest_config, require_freeze
from dogbreed.training.cached import load_split
from dogbreed.training.train_heads import load_head
from dogbreed.utils import auto_device, get_logger, save_json

log = get_logger()


@torch.no_grad()
def maps_for(head, data, device, batch=128):
    """-> dict with per-image map (n, N) of the decisive query, final logits, stage-1 logits, cand,
    and for re-rank heads the maps of all candidates (n, K, N)."""
    head.eval()
    out = {"map": [], "logits": [], "logits_global": [], "cand": [], "cand_maps": []}
    for s in range(0, data.n, batch):
        idx = np.arange(s, min(s + batch, data.n))
        g, tok, _ = data.batch(idx, device)
        o = head(g, tok, return_attn=True)
        z = o["logits"].float()
        a = o["attn"].float()
        if "cand" in o:                                    # re-rank head
            if a.dim() == 5:                               # pairwise: (B, H, K, K-1, N)
                a = a.mean(dim=3)
            a = a.mean(dim=1)                              # (B, K, N)
            cand = o["cand"]
            pos = z.gather(1, cand).argmax(1)
            m = a[torch.arange(len(idx), device=a.device), pos]
            out["cand"].append(cand.cpu().numpy())
            out["cand_maps"].append(a.cpu().numpy().astype(np.float16))
        elif a.shape[2] == z.shape[1]:                     # all-class attention: (B, H, C, N)
            pred = z.argmax(1)
            m = a.mean(dim=1)[torch.arange(len(idx), device=a.device), pred]
        else:                                              # attentive probe: (B, H, M, N)
            m = a.mean(dim=(1, 2))
        out["map"].append(m.cpu().numpy())
        out["logits"].append(z.cpu().numpy())
        if "logits_global" in o:
            out["logits_global"].append(o["logits_global"].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items() if v}


def box_cover(row, grid, img_size):
    res = {}
    for part in ("body", "head"):
        b = _box(row, part, int(row["img_w"]), int(row["img_h"]))
        b = box_to_input_frame(b, int(row["img_w"]), int(row["img_h"]), img_size) if b else None
        res[part] = box_token_coverage(b, grid, img_size)
    return res


def quant(m):
    m = np.asarray(m, dtype=np.float64)
    m = m / max(m.max(), 1e-12)
    return "".join("%x" % v for v in np.clip(np.round(m * 15), 0, 15).astype(int))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="tsinghua_main")
    ap.add_argument("--images", required=True)
    ap.add_argument("--tags", default="rr_rival,rr_rival_soft,rr_pairwise,rr_class,rr_center,attnprobe,classattn_all")
    ap.add_argument("--fig-tags", default="rr_rival,rr_class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-show", type=int, default=3)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    exp_dir = paths.EXPERIMENTS / a.experiment
    cfg = latest_config(exp_dir)
    require_freeze(exp_dir, [(t, a.seed, exp_dir / f"{t}_s{a.seed}") for t in a.tags.split(",")
                             if (exp_dir / f"{t}_s{a.seed}" / "best.pt").exists()])
    dev = auto_device(a.device)
    df, names = apply_class_subset(cfg, load_manifest(cfg.get("manifest") or paths.MANIFESTS / f"{cfg['dataset']}.csv"))
    size = cfg.get("img_size", 224)
    vdir = paths.feature_dir(cfg["dataset"], cfg["backbone"], size, "orig", cfg.get("feature_tag", ""))
    data = load_split([vdir], df, a.split, dev, token_dims=cfg.get("token_dims"))
    grid = int(round(np.sqrt(data.n_tokens)))
    rows = df.set_index("id").loc[data.ids].reset_index().to_dict("records")
    cover = [box_cover(r, grid, size) for r in rows]
    y = data.y.numpy()
    res, stats = {}, {}
    for tag in a.tags.split(","):
        d = exp_dir / f"{tag}_s{a.seed}"
        if not (d / "best.pt").exists():
            log.warning("no run %s: skipped", d.name)
            continue
        head, _ = load_head(d / "best.pt", dev)
        torch.manual_seed(a.seed)          # only the random control draws at inference
        r = maps_for(head, data, dev)
        res[tag] = r
        st = {}
        for part in ("body", "head"):
            mass, area = [], []
            for m, c in zip(r["map"], cover):
                if c[part] is None or c[part].sum() <= 0:
                    continue
                mass.append(float((m * c[part]).sum() / max(m.sum(), 1e-12)))
                area.append(float(c[part].mean()))
            mass, area = np.array(mass), np.array(area)
            st[part] = {"n": int(len(mass)), "mass_mean": float(mass.mean()), "area_mean": float(area.mean()),
                        "lift_mean": float((mass / np.maximum(area, 1e-6)).mean()),
                        "share_mass_gt_area": float((mass > area).mean())}
        top = r["logits"].argmax(1)
        st["top1"] = float((top == y).mean())
        m = r["map"] / np.maximum(r["map"].sum(1, keepdims=True), 1e-12)
        st["entropy_mean"] = float((-(m * np.log(np.maximum(m, 1e-12))).sum(1)).mean())
        st["entropy_uniform"] = float(np.log(m.shape[1]))
        stats[tag] = st
        log.info("%s: body mass %.3f (area %.3f, lift %.2f); head mass %.3f (area %.3f, lift %.2f)", tag,
                 st["body"]["mass_mean"], st["body"]["area_mean"], st["body"]["lift_mean"],
                 st["head"]["mass_mean"], st["head"]["area_mean"], st["head"]["lift_mean"])
    out = paths.RESULTS / a.experiment / "explain"
    out.mkdir(parents=True, exist_ok=True)
    save_json({"split": a.split, "seed": a.seed, "grid": grid, "stats": stats,
               "note": "boxes used for evaluation only"}, out / "box_stats.json")

    fig_tags = [t for t in a.fig_tags.split(",") if t in res and "cand" in res[t]]
    if not fig_tags:
        return 0
    r0 = res[fig_tags[0]]
    p0, p1 = r0["logits_global"].argmax(1), r0["logits"].argmax(1)
    kinds = {"fixed": (p0 != y) & (p1 == y), "broken": (p0 == y) & (p1 != y), "both right": (p0 == y) & (p1 == y)}
    order = np.argsort(data.ids)
    shown = []
    for k, mask in kinds.items():
        sel = [i for i in order if mask[i]][:a.n_show]
        shown += [(k, i) for i in sel]
        log.info("%s: %d images (%.2f %%), showing %d", k, int(mask.sum()), 100 * mask.mean(), len(sel))
    src = ImageSource(a.images)
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode
    crop = T.Compose([T.Resize(default_resize(size), interpolation=InterpolationMode.BICUBIC), T.CenterCrop(size)])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = 1 + 2 * len(fig_tags)
    fig, axes = plt.subplots(len(shown), ncol, figsize=(2.0 * ncol, 2.1 * len(shown)))
    axes = np.atleast_2d(axes)
    dump = []
    for row_i, (kind, i) in enumerate(shown):
        img = np.asarray(crop(src.open(rows[i]["key"]).convert("RGB")))
        ax = axes[row_i, 0]
        ax.imshow(img)
        ax.set_title(f"{kind}\ntrue: {names[y[i]][:22]}", fontsize=7)
        ent = {"id": int(data.ids[i]), "key": rows[i]["key"], "kind": kind, "true": names[y[i]], "tags": {}}
        for j, tag in enumerate(fig_tags):
            r = res[tag]
            cand = r["cand"][i]
            zc = r["logits"][i][cand]
            best = np.argsort(-zc)[:2]
            ent["tags"][tag] = {"cand": [names[c] for c in cand], "final_logit": np.round(zc, 3).tolist(),
                                "maps": [quant(r["cand_maps"][i][b]) for b in range(len(cand))]}
            for col, b in enumerate(best):
                ax = axes[row_i, 1 + 2 * j + col]
                ax.imshow(img)
                ax.imshow(r["cand_maps"][i][b].astype(np.float32).reshape(grid, grid), cmap="magma",
                          alpha=0.6, extent=(0, size, size, 0), interpolation="bilinear")
                mark = "✓" if cand[b] == y[i] else "✗"
                ax.set_title(f"{tag} #{col + 1}: {names[cand[b]][:18]} {mark}", fontsize=7)
        dump.append(ent)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"{a.experiment}, {a.split} split, seed {a.seed}: attention of the two strongest final candidates "
                 "(each map scaled to its own range; first images by id of each kind)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out / f"explain_{a.split}.png", dpi=110)
    save_json({"grid": grid, "img_size": size, "quantisation": "16 levels of map / max(map), hex, row-major",
               "images": dump}, out / "shown_maps.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
