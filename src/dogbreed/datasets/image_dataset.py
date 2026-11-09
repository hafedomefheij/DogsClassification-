"""A torch Dataset over manifest rows."""
from __future__ import annotations

from typing import Callable, Optional

import pandas as pd
import torch
from PIL import ImageOps
from torch.utils.data import Dataset

from . import corruptions
from .sources import ImageSource


class ManifestDataset(Dataset):
    """Yields (image tensor, label, manifest id).

    Args:
        df: manifest rows to serve (any split).
        source: where the images live (folder or zip).
        transform: torchvision transform applied last.
        corruption: optional name from ``corruptions.CORRUPTIONS`` (applied first, seeded by id).
        hflip: mirror every image (the second cached "view" for frozen-feature training).
    """

    def __init__(self, df: pd.DataFrame, source: ImageSource, transform: Callable,
                 corruption: Optional[str] = None, hflip: bool = False):
        self.df = df.reset_index(drop=True)
        self.keys = self.df["key"].tolist()
        self.labels = self.df["label"].astype(int).tolist()
        self.ids = self.df["id"].astype(int).tolist()
        self.source, self.transform = source, transform
        self.corruption, self.hflip = corruption, hflip
        self._rows = self.df.to_dict("records") if corruption else None

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, i: int):
        try:
            img = self.source.open(self.keys[i])
        except Exception as exc:  # surface the offending file instead of a bare worker crash
            raise RuntimeError(f"cannot decode {self.keys[i]}: {exc}") from exc
        if self.corruption:
            img = corruptions.apply(self.corruption, img, self._rows[i], seed=self.ids[i])
        if self.hflip:
            img = ImageOps.mirror(img)
        return self.transform(img), self.labels[i], self.ids[i]


def collate(batch):
    xs, ys, ids = zip(*batch)
    return torch.stack(xs), torch.tensor(ys), torch.tensor(ids)
