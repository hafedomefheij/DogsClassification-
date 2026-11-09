"""Provenance subsets of the test split, evaluated from saved test predictions."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dogbreed import paths
from dogbreed.utils import load_json, save_json

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location("subset_eval", ROOT / "scripts" / "subset_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pairs():
    # test ids 0..9: 0, 1, 2 have a confirmed copy; 3 a candidate only; 4 a candidate and a confirmed pair
    # (confirmed wins); 100 is a training image and must not matter
    return pd.DataFrame({"t_id": [0, 1, 2, 3, 4, 4, 100],
                         "confirmed": [True, True, True, False, False, True, True]})


def test_subsets_are_disjoint_and_cover_the_test_split():
    mod = _load_script()
    s = mod.provenance_subsets(_pairs(), np.arange(10)[::-1])
    assert s["stanford_origin"].tolist() == [0, 1, 2, 4]
    assert s["ambiguous"].tolist() == [3]
    assert s["independent"].tolist() == [5, 6, 7, 8, 9]
    assert sorted(np.concatenate([s["stanford_origin"], s["ambiguous"], s["independent"]]).tolist()) == list(range(10))
    # a CSV round trip stores the booleans as text
    p = _pairs().astype({"confirmed": str})
    assert mod.provenance_subsets(p, np.arange(10))["stanford_origin"].tolist() == [0, 1, 2, 4]


def test_end_to_end_from_saved_predictions(tmp_path, monkeypatch):
    mod = _load_script()
    for name in ("EXPERIMENTS", "RESULTS"):
        monkeypatch.setattr(paths, name, tmp_path / name.lower())
    ids = np.arange(10)
    y = np.zeros(10, dtype=np.int64)
    # A is right everywhere; B is right only on the independent images 5..9
    right = np.tile([5.0, 0.0], (10, 1))
    wrong = np.tile([0.0, 5.0], (10, 1))
    for tag, ok in (("A", np.ones(10, bool)), ("B", ids >= 5)):
        for seed in (0, 1):
            d = paths.EXPERIMENTS / "exp" / f"{tag}_s{seed}"
            d.mkdir(parents=True)
            (d / "best.pt").write_bytes(b"x")
            save_json({"config": {"lr": 1e-3}, "val_top1": 0.5}, d / "summary.json")
            perm = np.random.default_rng(seed).permutation(10)          # rows in any order
            logits = np.where(ok[:, None], right, wrong)
            np.savez(d / "test_orig.npz", ids=ids[perm], y=y[perm], logits=logits[perm])
    pairs_csv = tmp_path / "pairs.csv"
    _pairs().to_csv(pairs_csv, index=False)
    with pytest.raises(SystemExit, match="not frozen"):
        mod.main(["--experiment", "exp", "--pairs", str(pairs_csv)])
    spec = importlib.util.spec_from_file_location("freeze", ROOT / "scripts" / "freeze.py")
    fr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fr)
    assert fr.main(["--experiment", "exp", "--expect-seeds", "2"]) == 0
    assert mod.main(["--experiment", "exp", "--pairs", str(pairs_csv), "--compare", "A:B", "--n-boot", "200"]) == 0
    r = load_json(paths.RESULTS / "exp" / "orig" / "subsets.json")
    assert r["n"] == {"all": 10, "stanford_origin": 4, "independent": 5, "ambiguous": 1}
    assert r["top1"]["all"]["B"]["mean"] == 0.5
    assert r["top1"]["stanford_origin"]["B"]["mean"] == 0.0
    assert r["top1"]["independent"]["B"]["mean"] == 1.0
    assert r["top1"]["independent"]["A"]["n"] == 2
    so = r["comparisons"]["stanford_origin"]["A:B"]["per_seed"]
    assert [x["diff"] for x in so] == [1.0, 1.0] and so[0]["a_only"] == 4 and so[0]["b_only"] == 0
    assert r["comparisons"]["independent"]["A:B"]["mean_diff"] == 0.0
    assert "ambiguous" not in r["top1"]
