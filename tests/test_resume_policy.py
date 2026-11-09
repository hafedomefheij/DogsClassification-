"""The resume rule keeps finished runs only when the rebuilt cache leaves their validation top-1 in place."""
import importlib.util
from pathlib import Path

import pytest

from dogbreed import paths
from dogbreed.utils import load_json, save_json

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location("resume_policy", ROOT / "scripts" / "resume_policy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup(tmp_path, monkeypatch, diff):
    monkeypatch.setattr(paths, "EXPERIMENTS", tmp_path / "experiments")
    (paths.EXPERIMENTS / "exp" / "rr_rival_s0").mkdir(parents=True)
    drift = tmp_path / "drift" / "exp.json"
    save_json({"summary_excluding_random": {"runs": 3, "max_abs_top1_diff": diff}}, drift)
    return drift


def test_small_drift_keeps_the_finished_runs(tmp_path, monkeypatch):
    drift = _setup(tmp_path, monkeypatch, 0.0005)
    assert _load_script().main(["--experiment", "exp", "--drift", str(drift)]) == 0
    assert (paths.EXPERIMENTS / "exp" / "rr_rival_s0").exists()
    assert load_json(drift.with_name("exp_policy.json"))["decision"].startswith("keep")


def test_large_drift_sets_the_runs_aside(tmp_path, monkeypatch):
    drift = _setup(tmp_path, monkeypatch, 0.004)
    assert _load_script().main(["--experiment", "exp", "--drift", str(drift)]) == 0
    assert not (paths.EXPERIMENTS / "exp").exists()
    assert (paths.EXPERIMENTS / "exp_before_rebuild" / "rr_rival_s0").exists()
    rec = load_json(drift.with_name("exp_policy.json"))
    assert rec["decision"].startswith("set the finished runs aside") and rec["moved_to"].endswith("exp_before_rebuild")
    (paths.EXPERIMENTS / "exp").mkdir()
    with pytest.raises(SystemExit, match="already exists"):
        _load_script().main(["--experiment", "exp", "--drift", str(drift)])
