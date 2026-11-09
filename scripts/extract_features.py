"""Extract frozen-backbone features for every image of a manifest (resumable).

    python -I scripts/extract_features.py --dataset tsinghua --backbone dinov2_vitb14_reg \
        --images /content/data/tsinghua/low-resolution.zip --views orig --token-pca 128
    # Stanford tokens in the Tsinghua PCA basis (cross-dataset transfer), tag pca256_tsinghua:
    python -I scripts/extract_features.py --dataset stanford ... --token-pca 256 --pca-from tsinghua
    # test-time perturbations (test split only), same PCA:
    python -I scripts/extract_features.py ... --splits test --views blur,jpeg,occlude,bg_gray --token-pca 128

--token-pca D stores the patch tokens as their first D principal components. The PCA is fitted
once, on --pca-images random images of OUR TRAINING split only (never val or test), saved as
features/<dataset>/<backbone>_<size>_pca<D>/pca.npz and reused for every view. The global
descriptor g = [CLS; mean patch] is always kept at full width.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from dogbreed import paths
from dogbreed.datasets.corruptions import CORRUPTIONS
from dogbreed.datasets.manifest import load_manifest
from dogbreed.datasets.sources import ImageSource
from dogbreed.models.backbones import Backbone
from dogbreed.training.features import extract_features, fit_token_pca
from dogbreed.utils import auto_device, get_logger, save_json

log = get_logger()


def get_pca(df_all, src, bb, a, dev):
    if a.pca_from:  # another dataset's PCA (cross-dataset transfer: one token basis for both)
        f = paths.feature_root(a.pca_from, a.backbone, a.img_size, f"pca{a.token_pca}") / "pca.npz"
        if not f.exists():
            raise SystemExit(f"--pca-from {a.pca_from}: {f} not found; extract {a.pca_from} first")
        z = np.load(f)
        pca = {k: z[k] for k in z.files}
        log.info("using the %s token PCA %s (fitted on %s training images only)", a.pca_from, f,
                 a.pca_from)
        return pca
    root = paths.feature_root(a.dataset, a.backbone, a.img_size, f"pca{a.token_pca}")
    f = root / "pca.npz"
    if f.exists():
        z = np.load(f)
        pca = {k: z[k] for k in z.files}
        pca["explained_total"] = float(pca["explained_total"])
        pca["n_images_fit"] = int(pca["n_images_fit"])
        log.info("reusing token PCA %s (%.1f %% variance)", f, 100 * pca["explained_total"])
        return pca
    train = df_all[df_all["split"] == "train"]
    pca = fit_token_pca(train, src, bb, a.token_pca, n_images=a.pca_images, img_size=a.img_size,
                        batch_size=a.batch_size, workers=a.workers, device=str(dev), seed=0)
    root.mkdir(parents=True, exist_ok=True)
    np.savez(f, **{k: np.asarray(v) for k, v in pca.items()})
    save_json({"dim": a.token_pca, "explained_total": pca["explained_total"],
               "explained_cumulative": np.cumsum(pca["explained"]).tolist(),
               "n_images_fit": pca["n_images_fit"], "n_tokens_fit": pca["n_tokens_fit"],
               "fitted_on": "random subset of the training split (seed 0)"}, root / "pca.json")
    return pca


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--views", default="orig", help="comma list: orig, hflip or corruption names")
    ap.add_argument("--splits", default="train,val,test", help="which manifest splits to cache")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-tokens", action="store_true")
    ap.add_argument("--token-pca", type=int, default=0, help="store tokens as D principal components")
    ap.add_argument("--pca-images", type=int, default=4000)
    ap.add_argument("--pca-from", default=None,
                    help="reuse this dataset's token PCA (cache tag pca<D>_<dataset>)")
    a = ap.parse_args(argv)

    df_all = load_manifest(a.manifest or paths.MANIFESTS / f"{a.dataset}.csv")
    df = df_all[df_all["split"].isin(a.splits.split(","))]
    src = ImageSource(a.images)
    dev = auto_device(a.device)
    bb = Backbone(a.backbone, a.img_size)
    log.info("backbone %s", bb.describe())
    pca = get_pca(df_all, src, bb, a, dev) if a.token_pca else None
    tag = f"pca{a.token_pca}" if a.token_pca else ""
    if a.pca_from:
        tag += f"_{a.pca_from}"
    for view in a.views.split(","):
        corruption = view if view in CORRUPTIONS else None
        if view not in ("orig", "hflip") and corruption is None:
            raise SystemExit(f"unknown view {view!r}")
        out = paths.feature_dir(a.dataset, a.backbone, a.img_size, view, tag)
        extract_features(df, src, a.backbone, out, a.img_size, a.batch_size, a.workers, str(dev),
                         view=view, corruption=corruption, store_tokens=not a.no_tokens, bb=bb,
                         dataset=a.dataset, pca=pca)
    if dev.type == "cuda":
        log.info("peak GPU memory %.2f GB", torch.cuda.max_memory_allocated() / 2**30)
    return 0


if __name__ == "__main__":
    sys.exit(main())
