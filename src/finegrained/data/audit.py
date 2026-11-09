"""Integrity audit of one dataset: decodability, exact and near duplicates, label conflicts.

Signals per image (the pixel signals after the dataset's crop):
* ``sha1``: of the file bytes (exact copies);
* ``pix_sha1``: of the decoded RGB pixels (same picture, different file metadata);
* ``phash`` / ``dhash``: 64-bit perceptual hashes (Study 1's definitions), for near duplicates.
Embedding confirmation of near duplicates (cosine of frozen descriptors) happens later, once
features exist, with Study 1's thresholds; this stage only lists candidates, so no exclusion is
decided on hashes alone.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations_with_replacement
from typing import Dict

import numpy as np
import pandas as pd

from dogbreed.datasets.audit import (class_balance, dhash, duplicate_groups, hamming, near_duplicate_pairs,
                                     phash)

SPLITS = ("train", "val", "test")
PAIR_COLUMNS = ["id_a", "id_b", "split_a", "split_b", "label_a", "label_b", "phash_d", "dhash_d",
                "same_file", "same_pixels"]


def inspect_one(source, key: str) -> Dict:
    out = {"key": key, "ok": False, "error": "", "img_w": np.nan, "img_h": np.nan, "bytes": np.nan,
           "sha1": "", "pix_sha1": "", "phash": "", "dhash": ""}
    try:
        raw = source.read_bytes(key)
        out["bytes"] = len(raw)
        out["sha1"] = hashlib.sha1(raw).hexdigest()
        im = source.open(key)
        out["img_w"], out["img_h"] = im.size
        out["pix_sha1"] = hashlib.sha1(np.asarray(im, dtype=np.uint8).tobytes()).hexdigest()
        out["phash"] = f"{phash(im):016x}"
        out["dhash"] = f"{dhash(im):016x}"
        out["ok"] = True
    except Exception as exc:  # corrupt, truncated, unsupported
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return out


def hash_table(df: pd.DataFrame, source, workers: int = 8) -> pd.DataFrame:
    keys = df["key"].tolist()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(lambda k: inspect_one(source, k), keys, chunksize=64))
    h = pd.DataFrame(rows)
    h.insert(0, "id", df["id"].to_numpy())
    return h


def _u64(col: pd.Series) -> np.ndarray:
    return np.array([int(x, 16) if isinstance(x, str) and x else 0 for x in col], dtype=np.uint64)


def split_pair(a: str, b: str) -> str:
    order = {s: i for i, s in enumerate(SPLITS)}
    return "-".join(sorted((a, b), key=lambda s: order.get(s, 99)))


def candidate_pairs(df: pd.DataFrame, h: pd.DataFrame, max_phash: int = 8) -> pd.DataFrame:
    """Every pair of decodable images with pHash Hamming distance <= max_phash."""
    idx = np.flatnonzero(h["ok"].to_numpy())
    ph, dh = _u64(h["phash"].iloc[idx]), _u64(h["dhash"].iloc[idx])
    pairs = near_duplicate_pairs(ph, None, max_dist=max_phash)
    if not pairs:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    li = np.array([p[0] for p in pairs])
    lj = np.array([p[1] for p in pairs])
    i, j = idx[li], idx[lj]
    split = df["official_split"].to_numpy()
    label = df["label"].to_numpy()
    out = pd.DataFrame({
        "id_a": df["id"].to_numpy()[i], "id_b": df["id"].to_numpy()[j],
        "split_a": split[i], "split_b": split[j], "label_a": label[i], "label_b": label[j],
        "phash_d": np.array([p[2] for p in pairs]), "dhash_d": hamming(dh[li], dh[lj]),
        "same_file": h["sha1"].to_numpy()[i] == h["sha1"].to_numpy()[j],
        "same_pixels": h["pix_sha1"].to_numpy()[i] == h["pix_sha1"].to_numpy()[j]})
    return out.sort_values(["phash_d", "dhash_d", "id_a", "id_b"]).reset_index(drop=True)


def exact_groups(df: pd.DataFrame, h: pd.DataFrame, col: str) -> Dict:
    ok = h["ok"].to_numpy()
    codes = np.full(len(h), -1)
    codes[ok] = pd.factorize(h.loc[ok, col])[0]
    counts = np.bincount(codes[ok]) if ok.any() else np.array([], int)
    dup = np.flatnonzero(counts > 1)
    member = ok & np.isin(codes, dup)
    sub = pd.DataFrame({"code": codes[member], "split": df["official_split"].to_numpy()[member],
                        "label": df["label"].to_numpy()[member]})
    by, conflicts = {}, 0
    for _, g in sub.groupby("code"):
        key = "+".join(sorted(set(g["split"]), key=lambda s: SPLITS.index(s) if s in SPLITS else 99))
        by[key] = by.get(key, 0) + 1
        conflicts += int(g["label"].nunique() > 1)
    return {"groups": int(len(dup)), "images": int(member.sum()), "groups_by_splits": by,
            "groups_with_label_conflict": conflicts}


def summarise(df: pd.DataFrame, h: pd.DataFrame, pairs: pd.DataFrame) -> Dict:
    splits = [s for s in SPLITS if s in set(df["official_split"])]
    rep: Dict = {"images": int(len(df)), "decodable": int(h["ok"].sum()),
                 "errors": h.loc[~h["ok"], ["id", "error"]].to_dict("records")[:50],
                 "per_split": {s: int((df["official_split"] == s).sum()) for s in splits},
                 "classes": int(df["label"].nunique()),
                 "class_balance": {s: class_balance(df, s) for s in splits},
                 "image_size": {"w_median": float(h["img_w"].median()), "h_median": float(h["img_h"].median()),
                                "w_min": float(h["img_w"].min()), "h_min": float(h["img_h"].min())},
                 "exact_file": exact_groups(df, h, "sha1"),
                 "exact_pixels": exact_groups(df, h, "pix_sha1")}
    sp = (pairs.apply(lambda r: split_pair(r["split_a"], r["split_b"]), axis=1)
          if len(pairs) else pd.Series([], dtype=str))
    table = []
    for t in (0, 2, 4, 6, 8):
        sel = (pairs["phash_d"] <= t).to_numpy() if len(pairs) else np.zeros(0, bool)
        for a, b in combinations_with_replacement(splits, 2):
            m = sel & (sp == split_pair(a, b)).to_numpy() if len(pairs) else sel
            table.append({"phash_le": t, "splits": split_pair(a, b), "pairs": int(m.sum()),
                          "same_label": int((m & (pairs["label_a"] == pairs["label_b"]).to_numpy()).sum())
                          if len(pairs) else 0,
                          "also_dhash_le8": int((m & (pairs["dhash_d"] <= 8).to_numpy()).sum())
                          if len(pairs) else 0})
    rep["phash_candidates"] = table
    pos = {k: i for i, k in enumerate(df["id"].to_numpy())}
    close = pairs[pairs["phash_d"] <= 4] if len(pairs) else pairs
    groups = duplicate_groups(len(df), [(pos[a], pos[b]) for a, b in close[["id_a", "id_b"]].to_numpy()])
    rep["phash_le4_groups"] = int((pd.Series(groups).value_counts() > 1).sum())
    return rep
