"""The drift check compares a head's validation predictions on a rebuilt cache with the ones saved at training."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dogbreed import paths
from dogbreed.utils import load_json, save_json

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location("feature_drift_check", ROOT / "scripts" / "feature_drift_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup(tmp_path, monkeypatch, mod, now_logits):
    for name in ("EXPERIMENTS", "RESULTS", "MANIFESTS", "FEATURES"):
        monkeypatch.setattr(paths, name, tmp_path / name.lower())
    exp = paths.EXPERIMENTS / "exp"
    exp.mkdir(parents=True)
    save_json({"config": {"dataset": "toy", "backbone": "bb", "feature_tag": "pca256", "token_dims": 8}},
              exp / "config_20261108_000000.json")
    ids, y = np.arange(4), np.array([0, 1, 0, 1])
    then = np.array([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 0.0]])        # 3 of 4 right
    for tag, mode in (("rr_rival", "rival"), ("rr_random", "random")):
        d = exp / f"{tag}_s0"
        d.mkdir()
        (d / "best.pt").write_bytes(b"x")
        np.savez(d / "val_preds.npz", ids=ids, y=y, logits=then)
    monkeypatch.setattr(mod, "load_manifest", lambda p: None)
    monkeypatch.setattr(mod, "apply_class_subset", lambda cfg, df: (df, ["a", "b"]))
    monkeypatch.setattr(mod, "load_split", lambda dirs, df, split, device, token_dims=None:
                        SimpleNamespace(split=split, n=4))
    monkeypatch.setattr(mod, "load_head", lambda p, device: (
        None, {"config": {"head": {"mode": "random" if "random" in str(p) else "rival"}}}))
    monkeypatch.setattr(mod, "predict", lambda head, data, device: {"ids": ids, "y": y, "logits": now_logits})


def test_reports_changed_predictions_and_skips_the_random_control(tmp_path, monkeypatch):
    mod = _load_script()
    now = np.array([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.1]])         # last image flips: 4 of 4 right
    _setup(tmp_path, monkeypatch, mod, now)
    out = tmp_path / "drift.json"
    assert mod.main(["--experiment", "exp", "--out", str(out), "--device", "cpu"]) == 0
    r = load_json(out)
    assert r["split"] == "val" and r["n_images"] == 4
    rival = next(x for x in r["runs"] if x["tag"] == "rr_rival")
    assert rival["top1_then"] == 0.75 and rival["top1_now"] == 1.0
    assert rival["changed_images"] == 1 and rival["changed_share"] == 0.25
    assert rival["max_abs_logit_diff"] == pytest.approx(2.1)
    s = r["summary_excluding_random"]
    assert s["runs"] == 1 and s["max_abs_top1_diff"] == 0.25 and s["max_changed_share"] == 0.25


def test_refuses_when_the_validation_images_differ(tmp_path, monkeypatch):
    mod = _load_script()
    _setup(tmp_path, monkeypatch, mod, np.zeros((4, 2)))
    monkeypatch.setattr(mod, "predict", lambda head, data, device: {"ids": np.arange(1, 5), "y": np.zeros(4),
                                                                     "logits": np.zeros((4, 2))})
    with pytest.raises(SystemExit, match="validation images differ"):
        mod.main(["--experiment", "exp", "--out", str(tmp_path / "d.json"), "--device", "cpu"])
