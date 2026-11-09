"""Choose the held-out ("unknown") breeds for the open-set protocol, at random with a fixed seed.

    python -I scripts/make_openset_split.py --dataset tsinghua --unknown 30 --seed 0

Writes configs/openset/<dataset>_unknown<U>_s<seed>.json = {"known": [labels], "unknown": [labels],
"known_names": [...], "unknown_names": [...]}. Heads trained with ``class_subset`` pointing at this
file never see an image of an unknown breed (train or val); the test images of the unknown breeds
are the out-of-distribution set of scripts/ood_eval.py. The choice is random rather than hand-picked
so that it cannot favour any method. Caveat, reported with the results: the frozen backbone was
pretrained on data that contains these breeds, so they are unknown to the heads, not to the backbone.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from dogbreed import paths
from dogbreed.datasets.manifest import load_manifest
from dogbreed.utils import save_json


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tsinghua")
    ap.add_argument("--unknown", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    df = load_manifest(paths.MANIFESTS / f"{a.dataset}.csv")
    names = df.groupby("label")["class_name"].first().sort_index()
    labels = names.index.to_numpy()
    rng = np.random.default_rng(a.seed)
    unknown = np.sort(rng.choice(labels, size=a.unknown, replace=False))
    known = np.setdiff1d(labels, unknown)
    out = Path(a.out or Path("configs/openset") / f"{a.dataset}_unknown{a.unknown}_s{a.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json({"dataset": a.dataset, "seed": a.seed, "rule": "numpy default_rng(seed).choice, no replacement",
               "known": known.tolist(), "unknown": unknown.tolist(),
               "known_names": names[known].tolist(), "unknown_names": names[unknown].tolist()}, out)
    print(f"{out}: {len(known)} known, {len(unknown)} unknown breeds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
