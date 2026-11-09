"""Frozen-backbone feature extraction into an on-disk, resumable cache.

Cache layout (one folder per dataset x backbone x input size x view):
    meta.json     backbone, view/corruption, sizes, code hash
    ids.npy       int64 (N,)        row -> manifest id
    global.npy    float32 (N, Dg)   global descriptor g
    tokens.npy    float16 (N, T, D) spatial tokens (memory-mapped)
    done.npy      bool (N,)         rows already written (resume after an interruption)
"""
from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..datasets.image_dataset import ManifestDataset, collate
from ..datasets.sources import ImageSource
from ..datasets.transforms import eval_transform
from ..models.backbones import Backbone
from ..utils import autocast, code_hash, get_logger, load_json, save_json

log = get_logger()


class FeatureCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.meta = load_json(self.path / "meta.json")
        self.ids = np.load(self.path / "ids.npy")
        self._row = {int(i): r for r, i in enumerate(self.ids)}
        self._g = self._t = None

    # ---- creation / writing -------------------------------------------------------------
    @classmethod
    def create(cls, path, ids, dim_global, n_tokens, dim_token, meta, store_tokens=True):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        ids = np.asarray(ids, dtype=np.int64)
        if (path / "meta.json").exists() and (path / "ids.npy").exists():
            old = np.load(path / "ids.npy")
            if old.shape == ids.shape and (old == ids).all():
                return cls(path)                      # resume
            raise ValueError(f"{path} holds a cache for different ids; delete it or change the path")
        n = len(ids)
        np.save(path / "ids.npy", ids)
        np.lib.format.open_memmap(path / "global.npy", "w+", np.float32, (n, dim_global)).flush()
        if store_tokens:
            np.lib.format.open_memmap(path / "tokens.npy", "w+", np.float16, (n, n_tokens, dim_token)).flush()
        np.save(path / "done.npy", np.zeros(n, dtype=bool))
        meta = dict(meta, n=n, dim_global=dim_global, n_tokens=n_tokens, dim_token=dim_token,
                    store_tokens=store_tokens)
        save_json(meta, path / "meta.json")
        return cls(path)

    def open_for_write(self):
        self._wg = np.lib.format.open_memmap(self.path / "global.npy", "r+")
        self._wt = (np.lib.format.open_memmap(self.path / "tokens.npy", "r+")
                    if self.meta.get("store_tokens", True) else None)
        self._done = np.load(self.path / "done.npy")

    def write(self, rows: np.ndarray, g: np.ndarray, tokens: Optional[np.ndarray]):
        self._wg[rows] = g
        if self._wt is not None and tokens is not None:
            self._wt[rows] = tokens
        self._done[rows] = True

    def flush(self):
        self._wg.flush()
        if self._wt is not None:
            self._wt.flush()
        np.save(self.path / "done.npy", self._done)

    # ---- reading -----------------------------------------------------------------------
    @property
    def done(self) -> np.ndarray:
        return np.load(self.path / "done.npy")

    @property
    def complete(self) -> bool:
        return bool(self.done.all())

    def rows(self, ids) -> np.ndarray:
        try:
            return np.array([self._row[int(i)] for i in ids], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(f"id {exc} not in cache {self.path}") from None

    def global_feats(self) -> np.ndarray:
        if self._g is None:
            self._g = np.load(self.path / "global.npy", mmap_mode="r")
        return self._g

    def tokens(self) -> np.ndarray:
        if self._t is None:
            self._t = np.load(self.path / "tokens.npy", mmap_mode="r")
        return self._t


def _loader(df, source, bb, img_size, batch_size, workers, dev, view, corruption):
    tf = eval_transform(img_size, mean=bb.spec.mean, std=bb.spec.std)
    ds = ManifestDataset(df, source, tf, corruption, hflip=(view == "hflip"))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                      collate_fn=collate, pin_memory=dev.type == "cuda",
                      prefetch_factor=4 if workers else None, persistent_workers=False)


def _amp_ctx(dev, amp):
    return autocast(dev, training=False) if amp else nullcontext()


def fit_token_pca(df_train: pd.DataFrame, source: ImageSource, bb: Backbone, dim: int,
                  n_images: int = 4000, img_size: int = 224, batch_size: int = 64, workers: int = 2,
                  device: str = "cuda", seed: int = 0) -> dict:
    """PCA of the patch tokens, fitted on a random subset of TRAINING images only.

    Returns {'mean': (D,), 'components': (dim, D), 'explained': (dim,), 'explained_total'}.
    Projecting the cached tokens to ``dim`` principal components keeps the token grid intact
    and lets the whole training split sit in GPU memory; every head reads the same projected
    tokens, so the comparison stays fair (the global descriptor g is not projected).
    """
    dev = torch.device(device)
    bb = bb.eval().to(dev)
    sub = df_train.sample(n=min(n_images, len(df_train)), random_state=seed).sort_values("id")
    dl = _loader(sub, source, bb, img_size, batch_size, workers, dev, "orig", None)
    d = bb.dim_token
    s1 = torch.zeros(d, dtype=torch.float64, device=dev)
    s2 = torch.zeros(d, d, dtype=torch.float64, device=dev)
    n = 0
    with torch.inference_mode():
        for x, _, _ in dl:
            with _amp_ctx(dev, True):
                _, tok = bb(x.to(dev, non_blocking=True))
            t = tok.reshape(-1, d).double()
            s1 += t.sum(0)
            s2 += t.T @ t
            n += t.shape[0]
    mean = s1 / n
    cov = s2 / n - torch.outer(mean, mean)
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order], evecs[:, order]
    comp = evecs[:, :dim].T.contiguous()
    total = float(evals.clamp(min=0).sum())
    expl = (evals[:dim].clamp(min=0) / total).float().cpu().numpy()
    log.info("token PCA %d -> %d on %d tokens from %d train images: %.1f %% variance kept",
             d, dim, n, len(sub), 100 * expl.sum())
    return {"mean": mean.float().cpu().numpy(), "components": comp.float().cpu().numpy(),
            "explained": expl, "explained_total": float(expl.sum()), "n_tokens_fit": int(n),
            "n_images_fit": int(len(sub)), "seed": seed}


def extract_features(df: pd.DataFrame, source: ImageSource, backbone: str, out_dir: str | Path,
                     img_size: int = 224, batch_size: int = 64, workers: int = 4,
                     device: str = "cuda", view: str = "orig", corruption: Optional[str] = None,
                     store_tokens: bool = True, amp: bool = True, bb: Optional[Backbone] = None,
                     dataset: str = "", pca: Optional[dict] = None) -> FeatureCache:
    """Run the frozen backbone over ``df`` (any order) and fill the cache; resumable.

    pca: optional output of ``fit_token_pca``; tokens are then stored as PCA coordinates.
    """
    dev = torch.device(device)
    bb = bb or Backbone(backbone, img_size)
    bb = bb.eval().to(dev)
    bb.requires_grad_(False)
    df = df.sort_values("id").reset_index(drop=True)
    meta = {"dataset": dataset, "backbone": bb.describe(), "view": view, "corruption": corruption,
            "img_size": img_size, "code_hash": code_hash(), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "token_pca": None if pca is None else {"dim": int(pca["components"].shape[0]),
                                                   "explained_total": pca["explained_total"],
                                                   "n_images_fit": pca["n_images_fit"]}}
    dim_tok = bb.dim_token if pca is None else int(pca["components"].shape[0])
    cache = FeatureCache.create(out_dir, df["id"].to_numpy(), bb.dim_global, bb.n_tokens,
                                dim_tok, meta, store_tokens)
    if pca is not None:
        np.savez(Path(out_dir) / "pca.npz", **{k: np.asarray(v) for k, v in pca.items()})
        p_mean = torch.from_numpy(pca["mean"]).to(dev)
        p_comp = torch.from_numpy(pca["components"]).to(dev)
    todo = np.nonzero(~cache.done)[0]
    if len(todo) == 0:
        log.info("cache complete: %s", out_dir)
        return cache
    dl = _loader(df.iloc[todo], source, bb, img_size, batch_size, workers, dev, view, corruption)
    ctx = _amp_ctx(dev, amp)
    cache.open_for_write()
    t0, pos = time.perf_counter(), 0
    with torch.inference_mode():
        for bi, (x, _, bid) in enumerate(dl):
            rows = todo[pos:pos + len(bid)]
            pos += len(bid)
            if not (cache.ids[rows] == bid.numpy()).all():
                raise RuntimeError("row/id mismatch while writing the cache")
            with ctx:
                g, tok = bb(x.to(dev, non_blocking=True))
            if pca is not None:
                tok = (tok.float() - p_mean) @ p_comp.T
            cache.write(rows, g.float().cpu().numpy(),
                        tok.to(torch.float16).cpu().numpy() if store_tokens else None)
            if bi % 25 == 0:
                cache.flush()
                el = time.perf_counter() - t0
                log.info("%s %s %s: %d/%d images, %.1f img/s", dataset, backbone, view, pos,
                         len(todo), pos / max(el, 1e-6))
    cache.flush()
    log.info("done %s in %.0f s", out_dir, time.perf_counter() - t0)
    return FeatureCache(out_dir)
