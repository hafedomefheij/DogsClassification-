"""Manifest utilities: our validation split, subsets, class names, saving and loading.

The manifest is the single source of truth for which image is in which split. Every feature
cache row and every prediction file is keyed by the manifest ``id``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def make_val_split(df: pd.DataFrame, frac: float = 0.1, seed: int = 0,
                   min_per_class: int = 1, exclude: Optional[pd.Series] = None) -> pd.DataFrame:
    """Stratified validation split carved out of the official training images.

    Adds column ``split`` in {train, val, test, excluded}. Rows flagged in ``exclude`` (e.g.
    near-duplicates of test images) are set to ``excluded`` and never used for training.
    """
    df = df.copy()
    df["split"] = np.where(df["official_split"] == "test", "test", "train")
    if exclude is not None:
        df.loc[exclude.values & (df["split"] == "train").values, "split"] = "excluded"
    rng = np.random.default_rng(seed)
    for _, idx in df[df["split"] == "train"].groupby("label").groups.items():
        idx = np.array(sorted(idx))
        n_val = max(min_per_class, int(round(frac * len(idx))))
        n_val = min(n_val, len(idx) - 1)
        pick = rng.choice(idx, size=n_val, replace=False)
        df.loc[pick, "split"] = "val"
    return df


def subsample(df: pd.DataFrame, per_class: dict, seed: int = 0) -> pd.DataFrame:
    """Keep at most per_class[split] images per class in each split (smoke tests, few-shot)."""
    rng = np.random.default_rng(seed)
    keep = []
    for split, n in per_class.items():
        part = df[df["split"] == split]
        for _, g in part.groupby("label"):
            idx = g.index.to_numpy()
            keep.extend(rng.choice(idx, size=min(n, len(idx)), replace=False).tolist())
    return df.loc[sorted(keep)].copy()


def class_names(df: pd.DataFrame) -> List[str]:
    names = df.drop_duplicates("label").sort_values("label")["class_name"].tolist()
    if len(names) != df["label"].nunique() or sorted(df["label"].unique()) != list(range(len(names))):
        raise ValueError("labels must be 0..C-1 with one name each")
    return names


def save_manifest(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=True)
    if "id" not in df or df["id"].duplicated().any():
        raise ValueError(f"{path}: missing or duplicated ids")
    return df


def split_ids(df: pd.DataFrame, split: str) -> np.ndarray:
    return df.loc[df["split"] == split, "id"].to_numpy()


def summary(df: pd.DataFrame) -> dict:
    out = {"images": int(len(df)), "classes": int(df["label"].nunique())}
    if "split" in df:
        out["per_split"] = {k: int(v) for k, v in df["split"].value_counts().items()}
    counts = df[df.get("official_split", "train") == "train"]["label"].value_counts()
    if len(counts):
        out["train_per_class_min"] = int(counts.min())
        out["train_per_class_max"] = int(counts.max())
        out["imbalance_ratio"] = float(counts.max() / counts.min())
    return out
