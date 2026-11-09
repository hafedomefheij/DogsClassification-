"""The cross-dataset overlap search on a tiny synthetic pair of datasets with planted copies."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from dogbreed import paths
from dogbreed.training.features import FeatureCache
from dogbreed.utils import load_json

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location("cross_overlap", ROOT / "scripts" / "cross_overlap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _manifest(n, dataset, splits, classes, hashes):
    return pd.DataFrame({
        "id": np.arange(n), "dataset": dataset, "key": [f"{dataset}/{i}.jpg" for i in range(n)],
        "label": [0] * n, "class_name": classes, "official_split": splits, "split": splits,
        "present": True, "ok": True, "phash": [f"{h:016x}" for h in hashes]})


def _cache(root, dataset, ids, g):
    c = FeatureCache.create(paths.feature_dir(dataset, "bb", 224, "orig", "t"), ids, g.shape[1],
                            1, 1, {"dataset": dataset}, store_tokens=False)
    c.open_for_write()
    c.write(np.arange(len(ids)), g, None)
    c.flush()


def test_planted_copies_are_found_and_classified(tmp_path, monkeypatch):
    for name in ("MANIFESTS", "FEATURES", "RESULTS"):
        monkeypatch.setattr(paths, name, tmp_path / name.lower())
    paths.MANIFESTS.mkdir(parents=True)
    rng = np.random.default_rng(0)
    gs = rng.normal(size=(5, 16)).astype(np.float32)
    hs = [int(x) for x in rng.integers(0, 2**63, size=5)]
    # Tsinghua: t0 = exact copy of s0 (train), t1 = slightly perturbed copy of s3 (train),
    # t2 = perturbed copy of s4 but only in pixels -> embedding far (should NOT be confirmed),
    # t3, t4 unrelated (test split).
    gt = rng.normal(size=(5, 16)).astype(np.float32)
    gt[0] = gs[0]
    gt[1] = gs[3] + 0.01 * rng.normal(size=16)
    ht = [int(x) for x in rng.integers(0, 2**63, size=5)]
    ht[0] = hs[0]
    ht[1] = hs[3] ^ 0b101          # Hamming distance 2
    ht[2] = hs[4] ^ 0b1            # pHash close, embedding unrelated
    # plus one image per dataset that its split excluded, so it has no cached descriptor:
    # t5 is a pHash copy of s1, s5 a pHash copy of t3 (the case that once crashed the script)
    t = _manifest(6, "tsinghua", ["train", "train", "train", "test", "test", "train"],
                  ["Chihuahua", "Pug", "Beagle", "Pug", "Pug", "Pug"], ht + [hs[1] ^ 0b1])
    s = _manifest(6, "stanford", ["train", "test", "test", "test", "test", "train"],
                  ["Chihuahua", "Pug", "Pug", "pug", "Beagle", "Pug"], hs + [ht[3] ^ 0b11])
    t.loc[5, "split"] = "excluded"
    s.loc[5, "split"] = "excluded"
    t.to_csv(paths.MANIFESTS / "tsinghua.csv", index=False)
    s.to_csv(paths.MANIFESTS / "stanford.csv", index=False)
    _cache(tmp_path, "tsinghua", t["id"][:5], gt)
    _cache(tmp_path, "stanford", s["id"][:5], gs)

    mod = _load_script()
    assert mod.main(["--backbone", "bb", "--feature-tag", "t", "--device", "cpu"]) == 0
    rep = load_json(paths.RESULTS / "audit" / "cross_tsinghua_stanford.json")
    assert rep["confirmed_pairs"] == 2
    assert rep["confirmed_by_split"] == {"train|test": 1, "train|train": 1}
    assert rep["confirmed_same_breed_name"] == 1.0            # 'Pug' == 'pug' after normalising
    assert rep["breed_correspondence"]["Pug"]["stanford"] == "pug"
    assert rep["stanford_test_in_tsinghua_trainval"] == 1      # s3
    clean = load_json(paths.RESULTS / "audit" / "stanford_clean_test.json")["ids"]
    assert 3 not in clean and 4 not in clean and {1, 2} <= set(clean)
    pairs = pd.read_csv(paths.RESULTS / "audit" / "cross_pairs.csv")
    weak = pairs[(pairs.t_id == 2) & (pairs.s_id == 4)].iloc[0]
    assert weak.phash == 1 and not weak.confirmed
    assert rep["with_descriptor"]["tsinghua"] == 5 and rep["with_descriptor"]["stanford"] == 5
    nodesc = pairs[((pairs.t_id == 5) & (pairs.s_id == 1)) | ((pairs.t_id == 3) & (pairs.s_id == 5))]
    assert len(nodesc) == 2 and nodesc.cos.isna().all() and not nodesc.confirmed.any()
    assert set(nodesc.phash) == {1, 2}
