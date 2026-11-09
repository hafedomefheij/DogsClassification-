"""Cross-dataset overlap between Tsinghua Dogs and Stanford Dogs.

    python -I scripts/cross_overlap.py --backbone dinov2_vitb14_reg --feature-tag pca256 \
        --tsinghua-images /content/data/raw/tsinghua_dogs/low-resolution.zip \
        --stanford-images /content/data/raw/stanford_dogs/Images

Tsinghua Dogs re-uses Stanford Dogs photographs at a different resolution and encoding, so
byte hashes (SHA-1) cannot find them. Two independent signals are measured for every
Tsinghua image against every Stanford image:
  * pHash: Hamming distance of the 64-bit DCT hashes stored in the manifests;
  * embedding: cosine similarity of the frozen global descriptors g (same backbone and view
    for both datasets, from the feature caches).
A pair is CONFIRMED when both agree (pHash <= --phash-dist and cosine >= --cos-pair), or
when the cosine alone is overwhelming (>= --cos-alone). Sensitivity tables over several
thresholds are written too, plus montages of pairs per similarity band for a visual check.

Outputs (results/audit/):
  cross_tsinghua_stanford.json  counts by (Tsinghua split, Stanford split), label agreement,
                                the empirical breed correspondence, threshold sensitivity
  cross_pairs.csv               every candidate pair with both signals
  stanford_clean_test.json      Stanford test ids matching no Tsinghua train/val image
  cross_nn_hist.png, cross_montage_*.png
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch

from dogbreed import paths
from dogbreed.datasets.audit import hex_to_u64, popcount64
from dogbreed.datasets.manifest import load_manifest
from dogbreed.training.features import FeatureCache
from dogbreed.utils import auto_device, get_logger, save_json

log = get_logger()


def norm_name(s: str) -> str:
    return " ".join(str(s).lower().replace("_", " ").replace("-", " ").split())


def as_bool(col: pd.Series) -> pd.Series:
    """CSV booleans may come back as bool, str or NaN (NaN must count as False)."""
    return col.astype(str).str.lower().eq("true")


def global_feats(dataset, backbone, img_size, tag, df):
    """-> (rows of df that have a cached descriptor, their descriptors).

    The orig cache holds the train, val and test images only; an image the split excluded (a
    near-duplicate of a test image) has no descriptor and keeps the pHash signal alone."""
    cache = FeatureCache(paths.feature_dir(dataset, backbone, img_size, "orig", tag))
    idx = np.nonzero(np.isin(df["id"].to_numpy(), cache.ids))[0]
    rows = cache.rows(df["id"].to_numpy()[idx])
    return idx, np.asarray(cache.global_feats()[rows], dtype=np.float32)


def nn_cosine(a: np.ndarray, b: np.ndarray, device, keep: float, block: int = 4096):
    """Nearest neighbour of every row of a in b (cosine), plus all pairs >= keep."""
    ta = torch.nn.functional.normalize(torch.as_tensor(a, device=device), dim=1)
    tb = torch.nn.functional.normalize(torch.as_tensor(b, device=device), dim=1)
    best, arg, pairs = [], [], []
    for s in range(0, len(ta), block):
        sim = ta[s:s + block] @ tb.T
        v, j = sim.max(dim=1)
        best.append(v.cpu()); arg.append(j.cpu())
        ii, jj = torch.nonzero(sim >= keep, as_tuple=True)
        pairs += [(int(i) + s, int(k), float(sim[i, k])) for i, k in zip(ii.tolist(), jj.tolist())]
    return torch.cat(best).numpy(), torch.cat(arg).numpy(), pairs


def nn_phash(h_a: np.ndarray, h_b: np.ndarray, keep: int, block: int = 1024):
    """Nearest neighbour of every hash of h_a in h_b (Hamming), plus all pairs <= keep."""
    best = np.empty(len(h_a), dtype=np.int16)
    arg = np.empty(len(h_a), dtype=np.int64)
    pairs = []
    for s in range(0, len(h_a), block):
        d = popcount64(np.bitwise_xor(h_a[s:s + block, None], h_b[None, :]))
        arg[s:s + block] = d.argmin(axis=1)
        best[s:s + block] = d[np.arange(len(d)), arg[s:s + block]]
        ii, jj = np.nonzero(d <= keep)
        pairs += [(int(i) + s, int(j), int(d[i, j])) for i, j in zip(ii, jj)]
    return best, arg, pairs


def montage(pairs_df, t_src, s_src, path, title, n=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = pairs_df.head(n)
    if rows.empty:
        return
    fig, ax = plt.subplots(len(rows), 2, figsize=(5, 2.4 * len(rows)))
    ax = np.atleast_2d(ax)
    for r, (_, p) in enumerate(rows.iterrows()):
        for c, (src, key, lab) in enumerate(((t_src, p.t_key, p.t_class), (s_src, p.s_key, p.s_class))):
            ax[r, c].imshow(src.open(key).convert("RGB"))
            ax[r, c].set_title(f"{lab[:22]}\ncos {p.cos:.3f}  pHash {p.phash}", fontsize=7)
            ax[r, c].axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2_vitb14_reg")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--feature-tag", default="pca256")
    ap.add_argument("--tsinghua-images", default=None)
    ap.add_argument("--stanford-images", default=None)
    ap.add_argument("--phash-dist", type=int, default=8)
    ap.add_argument("--cos-pair", type=float, default=0.85)
    ap.add_argument("--cos-alone", type=float, default=0.97)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    dev = auto_device(a.device)
    out = paths.RESULTS / "audit"
    out.mkdir(parents=True, exist_ok=True)

    t = load_manifest(paths.MANIFESTS / "tsinghua.csv")
    s = load_manifest(paths.MANIFESTS / "stanford.csv")
    t = t[as_bool(t["present"]) & as_bool(t["ok"])].sort_values("id").reset_index(drop=True)
    s = s[as_bool(s["present"]) & as_bool(s["ok"])].sort_values("id").reset_index(drop=True)
    log.info("tsinghua %d images, stanford %d images", len(t), len(s))

    # 1. pHash nearest neighbours (all images, including the ones the split excluded)
    hb, hj, hpairs = nn_phash(hex_to_u64(t["phash"]), hex_to_u64(s["phash"]), keep=a.phash_dist + 4)
    # 2. embedding nearest neighbours (images that have a cached descriptor)
    ti, gt = global_feats("tsinghua", a.backbone, a.img_size, a.feature_tag, t)
    si, gs = global_feats("stanford", a.backbone, a.img_size, a.feature_tag, s)
    log.info("cached descriptors: tsinghua %d of %d, stanford %d of %d (the others: pHash only)",
             len(ti), len(t), len(si), len(s))
    cb, cj, cpairs = nn_cosine(gt, gs, dev, keep=min(a.cos_pair, 0.80))
    t_pos = {int(r): k for k, r in enumerate(ti)}      # row of t -> row of gt
    s_pos = {int(r): k for k, r in enumerate(si)}      # row of s -> row of gs

    # candidate pairs = union of both signals, each pair scored by both; keys are (row of t, row of s)
    cand = {}
    for i, j, d in hpairs:
        cand[(int(i), int(j))] = {"phash": d}
    for i, j, c in cpairs:
        cand.setdefault((int(ti[i]), int(si[j])), {})["cos"] = c
    rows = []
    gs_n = torch.nn.functional.normalize(torch.as_tensor(gs), dim=1)
    gt_n = torch.nn.functional.normalize(torch.as_tensor(gt), dim=1)
    hs_all, ht_all = hex_to_u64(s["phash"]), hex_to_u64(t["phash"])
    for (i, j), v in cand.items():
        if "cos" not in v and i in t_pos and j in s_pos:
            v["cos"] = float(gt_n[t_pos[i]] @ gs_n[s_pos[j]])
        if "phash" not in v:
            v["phash"] = int(popcount64(np.array([ht_all[i] ^ hs_all[j]], dtype=np.uint64))[0])
        rows.append({"t_id": int(t["id"][i]), "t_key": t["key"][i], "t_split": t["split"][i],
                     "t_class": t["class_name"][i], "s_id": int(s["id"][j]), "s_key": s["key"][j],
                     "s_split": s["official_split"][j], "s_class": s["class_name"][j],
                     "phash": v["phash"], "cos": v.get("cos", np.nan)})
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        log.warning("no candidate pairs at all")
        pairs = pd.DataFrame(columns=["t_id", "t_key", "t_split", "t_class", "s_id", "s_key",
                                      "s_split", "s_class", "phash", "cos"])
    pairs["same_name"] = [norm_name(x) == norm_name(y) for x, y in zip(pairs.t_class, pairs.s_class)]
    pairs["confirmed"] = (((pairs.phash <= a.phash_dist) & (pairs.cos >= a.cos_pair))
                          | (pairs.cos >= a.cos_alone))
    pairs = pairs.sort_values(["confirmed", "cos"], ascending=False)
    pairs.to_csv(out / "cross_pairs.csv", index=False)
    conf = pairs[pairs.confirmed]

    # the empirical breed correspondence: majority Stanford class per Tsinghua class
    corr = {}
    for tc, grp in conf.groupby("t_class"):
        cnt = Counter(grp.s_class)
        sc, n = cnt.most_common(1)[0]
        corr[tc] = {"stanford": sc, "pairs": int(n), "purity": round(n / len(grp), 4)}

    def split_table(df):
        return {f"{ts}|{ss}": int(n) for (ts, ss), n in
                df.groupby(["t_split", "s_split"]).size().items()}

    sens = []
    for pdist in (4, 6, 8, 10):
        for cmin in (0.80, 0.85, 0.90, 0.95):
            m = pairs[(pairs.phash <= pdist) & (pairs.cos >= cmin)]
            sens.append({"phash_le": pdist, "cos_ge": cmin, "pairs": int(len(m)),
                         "tsinghua_images": int(m.t_id.nunique()),
                         "stanford_images": int(m.s_id.nunique())})
    leaked_test = set(conf[conf.t_split.isin(["train", "val"]) & (conf.s_split == "test")].s_id)
    s_test = set(s[s.official_split == "test"]["id"])
    clean = sorted(s_test - set(pairs[pairs.t_split.isin(["train", "val"])].s_id))
    report = {
        "backbone": a.backbone, "feature_tag": a.feature_tag,
        "rule": {"phash_dist": a.phash_dist, "cos_pair": a.cos_pair, "cos_alone": a.cos_alone},
        "n_tsinghua": int(len(t)), "n_stanford": int(len(s)),
        "with_descriptor": {"tsinghua": int(len(ti)), "stanford": int(len(si)),
                            "note": "images outside the train/val/test split have no cached descriptor; "
                                    "their pairs carry pHash only and cannot be confirmed"},
        "candidate_pairs": int(len(pairs)), "confirmed_pairs": int(len(conf)),
        "confirmed_tsinghua_images": int(conf.t_id.nunique()),
        "confirmed_stanford_images": int(conf.s_id.nunique()),
        "confirmed_by_split": split_table(conf),
        "confirmed_same_breed_name": float(conf.same_name.mean()) if len(conf) else None,
        "breed_correspondence": corr,
        "stanford_test_in_tsinghua_trainval": len(leaked_test),
        "stanford_test_clean": len(clean),
        "nn_cos_quantiles": {q: float(np.quantile(cb, q)) for q in (0.5, 0.9, 0.99, 0.999)},
        "nn_phash_quantiles": {q: float(np.quantile(hb, q)) for q in (0.001, 0.01, 0.1, 0.5)},
        "sensitivity": sens,
    }
    save_json(report, out / "cross_tsinghua_stanford.json")
    save_json({"ids": clean, "rule": "Stanford test images with no candidate match (pHash <= "
               f"{a.phash_dist + 4} or cosine >= {min(a.cos_pair, 0.80)}) to any Tsinghua train or "
               "val image"}, out / "stanford_clean_test.json")
    log.info("confirmed %d pairs: %d Tsinghua / %d Stanford images; by split %s; same name %.3f",
             len(conf), report["confirmed_tsinghua_images"], report["confirmed_stanford_images"],
             report["confirmed_by_split"], report["confirmed_same_breed_name"] or float("nan"))
    log.info("Stanford test images also in Tsinghua train/val: %d; clean Stanford test: %d",
             len(leaked_test), len(clean))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax[0].hist(cb, bins=100, log=True); ax[0].set_xlabel("nearest Stanford image: cosine")
    ax[1].hist(hb, bins=range(0, 34), log=True); ax[1].set_xlabel("nearest Stanford image: pHash distance")
    for x in ax:
        x.set_ylabel("Tsinghua images")
    fig.tight_layout(); fig.savefig(out / "cross_nn_hist.png", dpi=100); plt.close(fig)
    if a.tsinghua_images and a.stanford_images:
        from dogbreed.datasets.sources import ImageSource
        ts, ss = ImageSource(a.tsinghua_images), ImageSource(a.stanford_images)
        rng = np.random.default_rng(0)
        for lo, hi in ((0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 0.97), (0.97, 1.01)):
            band = pairs[(pairs.cos >= lo) & (pairs.cos < hi)]
            if len(band):
                band = band.iloc[rng.permutation(len(band))]
                montage(band, ts, ss, out / f"cross_montage_cos{lo:.2f}.png",
                        f"cosine {lo:.2f}-{hi:.2f}: {len(band)} pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
