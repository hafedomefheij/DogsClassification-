"""Finding the finished runs of an experiment, rebuilding its label space, and the selection freeze."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from ..datasets.manifest import class_names
from ..utils import load_json

FREEZE = "freeze.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require_freeze(exp_dir: Path, runs: Sequence[Tuple[str, int, Path]]) -> dict:
    """Refuse to touch a test split before model selection is frozen (scripts/freeze.py).

    Every run about to be evaluated must be listed in experiments/<exp>/freeze.json with the SHA-256 its
    best.pt had when the freeze was written, so no checkpoint can change between selection and testing.
    """
    f = Path(exp_dir) / FREEZE
    if not f.exists():
        raise SystemExit(f"{Path(exp_dir).name}: no {FREEZE}. Model selection is not frozen: run scripts/freeze.py "
                         f"after training, record it in research_log/research_log.md, then evaluate.")
    fz = load_json(f)
    for _, _, d in runs:
        r = fz["runs"].get(Path(d).name)
        if r is None:
            raise SystemExit(f"{Path(d).name} is not in {f}: it was trained after the freeze")
        if sha256_file(Path(d) / "best.pt") != r["sha256_best_pt"]:
            raise SystemExit(f"{Path(d).name}: best.pt changed after the freeze")
    return fz


def latest_config(exp_dir: Path) -> dict:
    """The experiment config recorded by the last run_heads.py call (experiments/<exp>/config_*.json)."""
    cfgs = sorted(Path(exp_dir).glob("config_*.json"))
    if not cfgs:
        raise SystemExit(f"no config in {exp_dir}")
    return load_json(cfgs[-1])["config"]


def run_dirs(exp_dir: Path, tags: Optional[Sequence[str]] = None) -> List[Tuple[str, int, Path]]:
    """(tag, seed, dir) of every finished seed run (best.pt present); learning-rate grid runs excluded."""
    out = []
    for d in sorted(Path(exp_dir).iterdir()):
        if not d.is_dir() or not (d / "best.pt").exists() or "_lr" in d.name:
            continue
        tag, seed = d.name.rsplit("_s", 1)
        if tags and tag not in tags:
            continue
        out.append((tag, int(seed), d))
    return out


def apply_class_subset(cfg: dict, df: pd.DataFrame):
    """Restrict a manifest to the known classes of an open-set config and relabel them 0..K-1.

    Returns (df, names). Without ``class_subset`` the manifest is returned unchanged.
    """
    if cfg.get("class_subset"):
        keep = set(load_json(cfg["class_subset"])["known"])
        df = df[df["label"].isin(keep)].copy()
        df["label"] = df["label"].map({c: i for i, c in enumerate(sorted(keep))})
    return df, class_names(df)
