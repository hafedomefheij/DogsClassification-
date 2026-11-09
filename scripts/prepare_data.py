"""Build the manifest, our validation split and the audit report for one dataset.

    python -I scripts/prepare_data.py --dataset tsinghua --images data/raw/tsinghua_dogs/low-resolution.zip
    python -I scripts/prepare_data.py --dataset stanford --images data/raw/stanford_dogs/Images

Steps
 1. parse the official split lists and annotations -> one row per image;
 2. audit every image: decodable?, true size, SHA-1, pHash, dHash (exact / near duplicates);
 3. duplicate analysis: exact duplicates and pHash near-duplicates (Hamming <= 4) within the
    dataset; any train image that duplicates a test image is EXCLUDED from training;
 4. stratified validation split (10 % of the official training images, seed 0);
 5. write data/manifests/<dataset>.csv and results/audit/<dataset>.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dogbreed import paths
from dogbreed.datasets import audit, manifest, stanford, tsinghua
from dogbreed.datasets.sources import ImageSource
from dogbreed.utils import get_logger, save_json

log = get_logger()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tsinghua", "stanford"])
    ap.add_argument("--raw", default=None, help="raw folder (default data/raw/<dataset>_dogs)")
    ap.add_argument("--images", required=True, help="image folder or zip")
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--phash-dist", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="audit only the first N images (smoke tests)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    raw = Path(a.raw) if a.raw else paths.RAW / f"{a.dataset}_dogs"
    if a.dataset == "tsinghua":
        df = tsinghua.build_manifest(raw, a.annotations)
    else:
        df = stanford.build_manifest(raw)
    log.info("%s: %d images, %d classes", a.dataset, len(df), df["label"].nunique())

    src = ImageSource(a.images)
    present = np.array([k in src for k in df["key"]])
    if not present.all():
        log.warning("%d listed images are missing from %s", (~present).sum(), a.images)
    df["present"] = present
    if a.limit:                                   # smoke tests: k images per class from each split
        k = max(2, a.limit // (2 * df["label"].nunique()))
        df = df[df["present"]].groupby(["label", "official_split"]).head(k).sort_values("id").copy()
        df["id"] = np.arange(len(df))
        df = df.reset_index(drop=True)
    keys = df.loc[df["present"], "key"].tolist()
    info = audit.inspect_images(src, keys, workers=a.workers).set_index("key")
    df = df.join(info, on="key")
    df["ok"] = df["ok"].fillna(False).astype(bool)

    # ---- duplicates ---------------------------------------------------------------------
    rep = {"dataset": a.dataset, "images_listed": int(len(df)), "missing": int((~df["present"]).sum()),
           "undecodable": int((df["present"] & ~df["ok"]).sum()),
           "undecodable_examples": df.loc[df["present"] & ~df["ok"], ["key", "error"]].head(20).values.tolist(),
           "modes": df["mode"].value_counts().to_dict(),
           "class_balance_train": audit.class_balance(df, "train"),
           "class_balance_test": audit.class_balance(df, "test")}
    good = df[df["ok"]].reset_index()
    sha_groups = good.groupby("sha1")["index"].apply(list)
    exact = [g for g in sha_groups if len(g) > 1]
    h = audit.hex_to_u64(good["phash"])
    pairs = audit.near_duplicate_pairs(h, max_dist=a.phash_dist)
    near = [(int(good.loc[i, "index"]), int(good.loc[j, "index"]), d) for i, j, d in pairs]

    def kind(i, j):
        s = {df.at[i, "official_split"], df.at[j, "official_split"]}
        return "train-test" if s == {"train", "test"} else ("test-test" if s == {"test"} else "train-train")

    def same_label(i, j):
        return bool(df.at[i, "label"] == df.at[j, "label"])

    exact_pairs = [(g[0], g[k]) for g in exact for k in range(1, len(g))]
    rep["exact_duplicate_groups"] = len(exact)
    rep["exact_pairs_by_kind"] = pd.Series([kind(i, j) for i, j in exact_pairs]).value_counts().to_dict() if exact_pairs else {}
    rep["exact_pairs_label_conflicts"] = int(sum(not same_label(i, j) for i, j in exact_pairs))
    rep["phash_near_pairs"] = len(near)
    rep["phash_pairs_by_kind"] = pd.Series([kind(i, j) for i, j, _ in near]).value_counts().to_dict() if near else {}
    rep["phash_pairs_label_conflicts"] = int(sum(not same_label(i, j) for i, j, _ in near))
    conflicts = [(df.at[i, "key"], df.at[j, "key"], d) for i, j, d in near if not same_label(i, j)]
    rep["label_conflict_examples"] = conflicts[:30]

    # exclude train images that duplicate (exactly or nearly) any test image
    leak = set()
    for i, j in exact_pairs + [(i, j) for i, j, _ in near]:
        si, sj = df.at[i, "official_split"], df.at[j, "official_split"]
        if si != sj:
            leak.add(i if si == "train" else j)
    exclude = df.index.isin(sorted(leak)) | ~df["ok"]
    rep["train_excluded_as_test_duplicates"] = int(len(leak))
    df = manifest.make_val_split(df, a.val_frac, a.seed, exclude=pd.Series(exclude, index=df.index))
    df.loc[~df["ok"] & (df["split"] == "test"), "split"] = "excluded"
    rep["splits"] = df["split"].value_counts().to_dict()
    rep["dup_group"] = None
    groups = audit.duplicate_groups(len(df), exact_pairs + [(i, j) for i, j, _ in near])
    df["dup_group"] = groups

    out = Path(a.out) if a.out else paths.MANIFESTS / f"{a.dataset}.csv"
    manifest.save_manifest(df, out)
    save_json(rep, paths.RESULTS / "audit" / f"{a.dataset}.json")
    log.info("manifest -> %s; splits %s", out, rep["splits"])
    log.info("audit: undecodable %d, exact groups %d, near pairs %d (train-test %s), label conflicts %d",
             rep["undecodable"], rep["exact_duplicate_groups"], rep["phash_near_pairs"],
             rep["phash_pairs_by_kind"].get("train-test", 0), rep["phash_pairs_label_conflicts"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
