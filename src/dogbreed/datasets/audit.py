"""Dataset audit: decodability, exact and near duplicates, class balance.

* exact duplicates: SHA-1 of the file bytes;
* near duplicates: 64-bit perceptual hash (pHash: DCT of a 32x32 greyscale thumbnail, the
  8x8 lowest frequencies thresholded at their median) and difference hash (dHash), compared by
  Hamming distance; and, when backbone embeddings are available, cosine similarity of the
  global descriptors (catches crops, re-encodings and small edits that move the hashes);
* class balance: counts, imbalance ratio, Gini coefficient.
"""
from __future__ import annotations

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

from .sources import ImageSource


def _grey(img: Image.Image, size: Tuple[int, int]) -> np.ndarray:
    return np.asarray(img.convert("L").resize(size, Image.LANCZOS), dtype=np.float32)


def phash(img: Image.Image) -> int:
    d = dctn(_grey(img, (32, 32)), norm="ortho")[:8, :8]
    bits = (d > np.median(d)).reshape(-1)
    return int(np.packbits(bits).view(">u8")[0])


def dhash(img: Image.Image) -> int:
    g = _grey(img, (9, 8))
    bits = (g[:, 1:] > g[:, :-1]).reshape(-1)
    return int(np.packbits(bits).view(">u8")[0])


def popcount64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int64)
    b = x.view(np.uint8).reshape(*x.shape, 8)
    return np.unpackbits(b, axis=-1).sum(-1).astype(np.int64)


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return popcount64(np.bitwise_xor(np.asarray(a, np.uint64), np.asarray(b, np.uint64)))


def inspect_one(source: ImageSource, key: str) -> Dict:
    out = {"key": key, "ok": False, "error": "", "img_w": np.nan, "img_h": np.nan,
           "bytes": np.nan, "sha1": "", "phash": "", "dhash": "", "mode": ""}
    try:
        raw = source.read_bytes(key)
        out["bytes"] = len(raw)
        out["sha1"] = hashlib.sha1(raw).hexdigest()
        with Image.open(io.BytesIO(raw)) as im:
            out["mode"] = im.mode
            im.load()
            rgb = im.convert("RGB")
        out["img_w"], out["img_h"] = rgb.size
        out["phash"] = f"{phash(rgb):016x}"
        out["dhash"] = f"{dhash(rgb):016x}"
        out["ok"] = True
    except Exception as exc:  # corrupt / truncated / unsupported
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return out


def inspect_images(source: ImageSource, keys: Iterable[str], workers: int = 4,
                   progress: bool = True) -> pd.DataFrame:
    keys = list(keys)
    it = None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        it = ex.map(lambda k: inspect_one(source, k), keys, chunksize=64)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, total=len(keys), desc="audit", mininterval=10)
        rows = list(it)
    return pd.DataFrame(rows)


def hex_to_u64(col: pd.Series) -> np.ndarray:
    return np.array([int(h, 16) if isinstance(h, str) and h else 0 for h in col], dtype=np.uint64)


def near_duplicate_pairs(h_a: np.ndarray, h_b: Optional[np.ndarray] = None, max_dist: int = 4,
                         block: int = 2048) -> List[Tuple[int, int, int]]:
    """All (i, j, distance) with Hamming(h_a[i], h_b[j]) <= max_dist (i < j when b is None)."""
    same = h_b is None
    h_b = h_a if same else h_b
    pairs = []
    for s in range(0, len(h_a), block):
        blk = h_a[s:s + block]
        d = popcount64(np.bitwise_xor(blk[:, None], h_b[None, :]))
        ii, jj = np.nonzero(d <= max_dist)
        for i, j in zip(ii + s, jj):
            if same and j <= i:
                continue
            pairs.append((int(i), int(j), int(d[i - s, j])))
    return pairs


def embedding_duplicate_pairs(emb_a, emb_b=None, threshold: float = 0.95,
                              block: int = 4096) -> List[Tuple[int, int, float]]:
    """Cosine-similarity pairs above ``threshold`` (torch matmul in blocks)."""
    import torch
    a = torch.nn.functional.normalize(torch.as_tensor(np.asarray(emb_a), dtype=torch.float32), dim=1)
    same = emb_b is None
    b = a if same else torch.nn.functional.normalize(
        torch.as_tensor(np.asarray(emb_b), dtype=torch.float32), dim=1)
    pairs = []
    for s in range(0, a.shape[0], block):
        sim = a[s:s + block] @ b.T
        ii, jj = torch.nonzero(sim >= threshold, as_tuple=True)
        for i, j in zip(ii.tolist(), jj.tolist()):
            gi = i + s
            if same and j <= gi:
                continue
            pairs.append((gi, j, float(sim[i, j])))
    return pairs


def duplicate_groups(n: int, pairs: Iterable[Tuple[int, int]]) -> np.ndarray:
    """Union-find over pairs -> group id per item (singletons get their own id)."""
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, *_ in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)])


def class_balance(df: pd.DataFrame, split: str = "train") -> Dict:
    counts = df[df["official_split"] == split]["label"].value_counts().sort_index().to_numpy()
    sorted_c = np.sort(counts)
    n = len(sorted_c)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(sorted_c) / (n * sorted_c.sum()))
    return {"split": split, "classes": int(n), "min": int(counts.min()), "max": int(counts.max()),
            "median": float(np.median(counts)), "imbalance_ratio": float(counts.max() / counts.min()),
            "gini": gini}
