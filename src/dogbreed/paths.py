"""Project paths. Every location can be moved with an environment variable (Colab uses this to
keep raw data and results on Google Drive and the large feature caches on the local disk)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("DOGBREED_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("DOGBREED_DATA", ROOT / "data"))
RAW = Path(os.environ.get("DOGBREED_RAW", DATA / "raw"))
MANIFESTS = Path(os.environ.get("DOGBREED_MANIFESTS", DATA / "manifests"))
FEATURES = Path(os.environ.get("DOGBREED_FEATURES", ROOT / "features"))
EXPERIMENTS = Path(os.environ.get("DOGBREED_EXPERIMENTS", ROOT / "experiments"))
RESULTS = Path(os.environ.get("DOGBREED_RESULTS", ROOT / "results"))
FIGURES = Path(os.environ.get("DOGBREED_FIGURES", ROOT / "figures"))
CHECKPOINTS = Path(os.environ.get("DOGBREED_CHECKPOINTS", ROOT / "checkpoints"))


def feature_dir(dataset: str, backbone: str, img_size: int, view: str = "orig", tag: str = "") -> Path:
    """features/<dataset>/<backbone>_<size>[_<tag>]/<view>.

    view is 'orig', 'hflip' or a corruption; tag names a token encoding such as 'pca128'.
    """
    return FEATURES / dataset / (f"{backbone}_{img_size}" + (f"_{tag}" if tag else "")) / view


def feature_root(dataset: str, backbone: str, img_size: int, tag: str = "") -> Path:
    return feature_dir(dataset, backbone, img_size, "orig", tag).parent
