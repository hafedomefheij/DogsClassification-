"""Load the cached features of one split into RAM (or GPU memory) and serve mini-batches.

A split may have several *views* (e.g. 'orig' and 'hflip' caches of the same images); during
training each sample draws one view at random, which acts as data augmentation without
running the backbone again.

Loading streams the memory-mapped cache in chunks straight into the destination (GPU memory
when it fits), so peak RAM stays at one chunk instead of two full copies of the split.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .features import FeatureCache


@dataclass
class SplitData:
    ids: np.ndarray                 # manifest ids, in row order
    y: torch.Tensor                 # (N,) long
    g: List[torch.Tensor]           # per view (N, Dg) float32
    tok: List[torch.Tensor]         # per view (N, T, D) float16 (RAM or device)
    device: torch.device

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def dim_global(self) -> int:
        return self.g[0].shape[1]

    @property
    def n_tokens(self) -> int:
        return self.tok[0].shape[1]

    @property
    def dim_token(self) -> int:
        return self.tok[0].shape[2]

    def batch(self, idx: np.ndarray, device: torch.device, generator: Optional[torch.Generator] = None,
              random_view: bool = False):
        it = torch.as_tensor(idx, dtype=torch.long)
        if random_view and len(self.tok) > 1:
            v = torch.randint(0, len(self.tok), (len(idx),), generator=generator)
            g, tok = _gather_views(self.g, it, v), _gather_views(self.tok, it, v)
        else:
            g = self.g[0][it.to(self.g[0].device)]
            tok = self.tok[0][it.to(self.tok[0].device)]
        y = self.y[it]
        return (g.to(device, non_blocking=True).float(), tok.to(device, non_blocking=True).float(),
                y.to(device, non_blocking=True))


def _gather_views(arrs: List[torch.Tensor], it: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Row it[i] of view v[i], vectorised (works for tensors in RAM or on the GPU)."""
    dev = arrs[0].device
    it, v = it.to(dev), v.to(dev)
    out = arrs[0][it]                       # advanced indexing: already a copy
    for k in range(1, len(arrs)):
        m = v == k
        if bool(m.any()):
            out[m] = arrs[k][it[m]]
    return out


def _read_rows(mm: np.ndarray, rows: np.ndarray, dtype: torch.dtype, device: torch.device,
               chunk: int = 2048, last_dims: Optional[int] = None) -> torch.Tensor:
    """mm[rows] -> tensor on ``device``, reading the memmap in sorted order, one chunk at a time.

    last_dims keeps only the first k entries of the last axis (the leading PCA components).
    """
    tail = tuple(mm.shape[1:])
    if last_dims:
        tail = tail[:-1] + (min(last_dims, tail[-1]),)
    out = torch.empty((len(rows),) + tail, dtype=dtype, device=device)
    order = np.argsort(rows, kind="stable")
    for s in range(0, len(rows), chunk):
        o = order[s:s + chunk]
        blk = mm[rows[o]]
        if last_dims:
            blk = blk[..., :tail[-1]]
        block = torch.from_numpy(np.ascontiguousarray(blk)).to(dtype)
        out[torch.from_numpy(o).to(device)] = block.to(device, non_blocking=False)
    return out


def load_split(cache_dirs: Sequence[str | Path], df: pd.DataFrame, split: str,
               device: torch.device, tokens_on_device: str = "auto",
               max_device_fraction: float = 0.6, token_dims: Optional[int] = None) -> SplitData:
    """Read the rows of ``split`` from every view cache.

    tokens_on_device: 'auto' (if all views fit in ``max_device_fraction`` of the free GPU
    memory), 'yes' or 'no'. On CPU the tokens always stay in RAM.
    token_dims: keep the first k token channels only (PCA caches: the k leading components,
    so one 256-component cache serves every smaller PCA width).
    """
    part = df[df["split"] == split].sort_values("id")
    ids = part["id"].to_numpy()
    y = torch.as_tensor(part["label"].to_numpy(), dtype=torch.long)
    caches = []
    for d in cache_dirs:
        cache = FeatureCache(d)
        if not cache.complete:
            raise RuntimeError(f"cache {d} is incomplete; finish extraction first")
        caches.append(cache)
    target = torch.device("cpu")
    if device.type == "cuda" and tokens_on_device != "no":
        need = sum(len(ids) * c.tokens().shape[1] * min(token_dims or 10**9, c.tokens().shape[2]) * 2
                   for c in caches)
        free, _ = torch.cuda.mem_get_info()
        if tokens_on_device == "yes" or need < max_device_fraction * free:
            target = device
    gs, toks = [], []
    for cache in caches:
        rows = cache.rows(ids)
        gs.append(_read_rows(cache.global_feats(), rows, torch.float32, target))
        toks.append(_read_rows(cache.tokens(), rows, torch.float16, target, last_dims=token_dims))
    return SplitData(ids=ids, y=y, g=gs, tok=toks, device=device)
