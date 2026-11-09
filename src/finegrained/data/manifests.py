"""Manifests for CUB-200-2011 and FGVC-Aircraft, built from the official list files only.

A manifest is a table with one row per image: ``id`` (stable, dataset-prefixed), ``key`` (path
relative to the image root, readable by ``dogbreed.datasets.sources.ImageSource``), ``label``
(0..C-1), ``class_name``, ``official_split`` and ``split`` (train / val / test / excluded).
Annotations that a model must never see at test time (bounding boxes, coarser labels) are kept in
separate columns for analysis only.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from dogbreed.datasets.manifest import make_val_split


def manifest_sha256(df: pd.DataFrame) -> str:
    """Hash of the columns that define a split (id, key, label, split): names the split in every record."""
    cols = [c for c in ("id", "key", "label", "split") if c in df]
    blob = df[cols].to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(blob).hexdigest()


def _find(root: Path, marker: str) -> Path:
    root = Path(root)
    if (root / marker).exists():
        return root
    hits = sorted(p.parent for p in root.rglob(marker))
    if not hits:
        raise FileNotFoundError(f"{marker} not found under {root}")
    return hits[0]


def _pairs(path: Path) -> Iterable[tuple]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            a, _, b = line.partition(" ")
            yield a, b.strip()


# ----------------------------------------------------------------------------------------- CUB
def cub_raw(root: str | Path) -> tuple[pd.DataFrame, Path]:
    """-> (manifest without a validation split, image root). ``root`` holds CUB_200_2011/."""
    base = _find(Path(root), "train_test_split.txt")
    images = dict(_pairs(base / "images.txt"))
    labels = dict(_pairs(base / "image_class_labels.txt"))
    is_train = dict(_pairs(base / "train_test_split.txt"))
    classes = dict(_pairs(base / "classes.txt"))
    boxes = {}
    if (base / "bounding_boxes.txt").exists():
        for line in (base / "bounding_boxes.txt").read_text().split("\n"):
            parts = line.split()
            if len(parts) == 5:
                boxes[parts[0]] = [float(v) for v in parts[1:]]
    rows = []
    for iid in sorted(images, key=int):
        cid = int(labels[iid])
        name = classes[str(cid)].split(".", 1)[-1]
        x, y, w, h = boxes.get(iid, [np.nan] * 4)
        rows.append({"id": f"cub_{int(iid):05d}", "key": images[iid], "label": cid - 1, "class_name": name,
                     "official_split": "train" if is_train[iid] == "1" else "test",
                     "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h})
    df = pd.DataFrame(rows)
    _check(df, 200)
    return df, base / "images"


# ------------------------------------------------------------------------------------ Aircraft
def aircraft_raw(root: str | Path) -> tuple[pd.DataFrame, Path]:
    """-> (manifest with the official train / val / test split, image root). Variant level."""
    data = _find(Path(root), "variants.txt")
    variants = [v.strip() for v in (data / "variants.txt").read_text(encoding="utf-8").splitlines() if v.strip()]
    vidx = {v: i for i, v in enumerate(variants)}
    boxes = {}
    if (data / "images_box.txt").exists():
        for a, b in _pairs(data / "images_box.txt"):
            boxes[a] = [float(v) for v in b.split()]
    rows = []
    for split in ("train", "val", "test"):
        var = dict(_pairs(data / f"images_variant_{split}.txt"))
        fam = dict(_pairs(data / f"images_family_{split}.txt"))
        man = dict(_pairs(data / f"images_manufacturer_{split}.txt"))
        for iid in sorted(var):
            x0, y0, x1, y1 = boxes.get(iid, [np.nan] * 4)
            rows.append({"id": f"air_{iid}", "key": f"images/{iid}.jpg", "label": vidx[var[iid]],
                         "class_name": var[iid], "family": fam.get(iid, ""), "manufacturer": man.get(iid, ""),
                         "official_split": split, "split": split,
                         "bbox_x0": x0, "bbox_y0": y0, "bbox_x1": x1, "bbox_y1": y1})
    df = pd.DataFrame(rows)
    _check(df, len(variants))
    return df, data


def _check(df: pd.DataFrame, n_classes: int) -> None:
    if df["id"].duplicated().any():
        raise ValueError("duplicated ids")
    if sorted(df["label"].unique()) != list(range(n_classes)):
        raise ValueError(f"labels are not 0..{n_classes - 1}")
    if (df.groupby("label")["class_name"].nunique() != 1).any():
        raise ValueError("a label with two class names")


def finalise(df: pd.DataFrame, val_from_train: Optional[float], seed: int,
             exclude_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Apply the exclusion list and the validation split.

    ``exclude_ids``: training (and validation) images that duplicate a test image; they are marked
    ``excluded`` and never used for training or selection. Test images are never excluded.
    """
    exclude = pd.Series(df["id"].isin(set(exclude_ids or ())).to_numpy(), index=df.index)
    if val_from_train is not None:
        out = make_val_split(df, frac=val_from_train, seed=seed, exclude=exclude)
    else:
        out = df.copy()
        out["split"] = out["official_split"]
        out.loc[exclude.to_numpy() & (out["split"] != "test").to_numpy(), "split"] = "excluded"
    return out


def split_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {k: int(v) for k, v in df["split"].value_counts().sort_index().items()}
