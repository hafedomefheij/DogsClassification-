"""The selection freeze that every test-reading script requires."""
import importlib.util
from pathlib import Path

import pytest

from dogbreed import paths
from dogbreed.evaluation.runs import require_freeze, run_dirs
from dogbreed.utils import load_json, save_json

ROOT = Path(__file__).resolve().parents[1]


def _freeze_script():
    spec = importlib.util.spec_from_file_location("freeze", ROOT / "scripts" / "freeze.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _runs(exp, tags=("linear", "rr_rival"), seeds=(0, 1, 2)):
    for t in tags:
        for s in seeds:
            d = exp / f"{t}_s{s}"
            d.mkdir(parents=True)
            (d / "best.pt").write_bytes(f"{t}{s}".encode())
            save_json({"config": {"lr": 1e-3}, "val_top1": 0.5 + s / 100, "val_nll": 1.0, "best_epoch": 3}, d / "summary.json")
        g = exp / f"{t}_lr0.001_s0"                  # a learning-rate grid run is never evaluated or frozen
        g.mkdir()
        (g / "best.pt").write_bytes(b"grid")


def test_freeze_then_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "EXPERIMENTS", tmp_path)
    exp = tmp_path / "e"
    _runs(exp)
    save_json({"linear": 1e-3, "rr_rival": 1e-3}, exp / "lr_selection.json")
    runs = run_dirs(exp)
    with pytest.raises(SystemExit, match="not frozen"):
        require_freeze(exp, runs)
    fr = _freeze_script()
    assert fr.main(["--experiment", "e"]) == 0
    fz = load_json(exp / "freeze.json")
    assert sorted(fz["runs"]) == sorted(d.name for _, _, d in runs) and len(fz["runs"]) == 6
    assert fz["lr_selection"] == {"linear": 1e-3, "rr_rival": 1e-3}
    assert require_freeze(exp, runs)["experiment"] == "e"
    # a second freeze is refused without --force
    with pytest.raises(SystemExit, match="exists"):
        fr.main(["--experiment", "e"])
    # a checkpoint that changes after the freeze is refused
    (exp / "rr_rival_s1" / "best.pt").write_bytes(b"retrained")
    with pytest.raises(SystemExit, match="changed after the freeze"):
        require_freeze(exp, runs)
    # a run trained after the freeze is refused
    (exp / "rr_rival_s1" / "best.pt").write_bytes(b"rr_rival1")
    d = exp / "rr_class_s0"
    d.mkdir()
    (d / "best.pt").write_bytes(b"new")
    with pytest.raises(SystemExit, match="after the freeze"):
        require_freeze(exp, run_dirs(exp))


def test_freeze_needs_all_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "EXPERIMENTS", tmp_path)
    _runs(tmp_path / "e", tags=("linear",), seeds=(0, 1))
    with pytest.raises(SystemExit, match="fewer than 3"):
        _freeze_script().main(["--experiment", "e"])
    assert _freeze_script().main(["--experiment", "e", "--expect-seeds", "2"]) == 0
