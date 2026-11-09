"""Checkpoints and prediction arrays from an archive whose own hash is not the recorded one are merged file by file,
each against the first session's records, and nothing is written when any check fails."""
import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("merge_rerun_files", ROOT / "scripts" / "merge_rerun_files.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _npz(**arrays):
    b = io.BytesIO()
    np.savez_compressed(b, **arrays)
    return b.getvalue()


def _setup(tmp_path, mod, weights=b"head-A", tamper=None):
    rng = np.random.default_rng(0)
    yv, yt = np.array([0, 1, 2, 0, 1, 2, 0, 1]), np.array([0, 1, 2, 2, 1, 0, 0, 2])
    val = {"ids": np.arange(8), "y": yv, "logits": rng.normal(size=(8, 3)).astype(np.float32)}
    test = {"ids": np.arange(8), "y": yt, "logits": rng.normal(size=(8, 3)).astype(np.float32)}
    grid_val = {"ids": np.arange(8), "y": yv, "logits": rng.normal(size=(8, 3)).astype(np.float32)}
    files = {"experiments/exp/linear_s0/best.pt": weights,
             "experiments/exp/linear_s0/val_preds.npz": _npz(**val),
             "experiments/exp/linear_s0/test_orig.npz": _npz(**test),
             "experiments/exp/linear_lr0.001_s0/best.pt": b"grid run at another rate",
             "experiments/exp/linear_lr0.001_s0/val_preds.npz": _npz(**grid_val)}
    archive = tmp_path / "full.tar"
    with tarfile.open(archive, "w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mtime = len(data), 1_700_000_000
            tar.addfile(info, io.BytesIO(data))
    summary, report, grid_summary = mod.val_metrics(val), mod.test_report(test, val, 3), mod.val_metrics(grid_val)
    if tamper:
        tamper(summary, report, grid_summary)
    records = {"experiments/exp/freeze.json": {"runs": {"linear_s0": {"sha256_best_pt":
                                                                      hashlib.sha256(b"head-A").hexdigest()}}},
               "experiments/exp/linear_s0/summary.json": summary,
               "experiments/exp/linear_s0/test_metrics_orig.json": {"tag": "linear", "top1_draws": [0.1], **report},
               "experiments/exp/linear_lr0.001_s0/summary.json": grid_summary}
    for dest in ("work", "mirror"):
        for rel, obj in records.items():
            p = tmp_path / dest / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(obj), encoding="utf-8")
    args = ["--archive", str(archive), "--work", str(tmp_path / "work"), "--mirror", str(tmp_path / "mirror"),
            "--record", str(tmp_path / "record")]
    return args, files


def _arrays_in(tmp_path):
    return sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")
                  if p.suffix in (".pt", ".npz") and "record" not in p.parts)


def test_checked_files_are_merged_with_a_manifest(tmp_path):
    mod = _script()
    args, files = _setup(tmp_path, mod)
    assert mod.main(args) == 0
    for dest in ("work", "mirror"):
        for rel, data in files.items():
            p = tmp_path / dest / rel
            assert p.read_bytes() == data
            assert int(p.stat().st_mtime) == 1_700_000_000
    manifest = next((tmp_path / "record").glob("merge_rerun_*.tsv")).read_text(encoding="utf-8").splitlines()
    assert len(manifest) == 1 + len(files)
    basis = {line.split("\t")[0]: line.split("\t")[3] for line in manifest[1:]}
    assert basis["experiments/exp/linear_s0/best.pt"] == "freeze.json"
    assert basis["experiments/exp/linear_lr0.001_s0/best.pt"] == "no recorded hash"
    assert basis["experiments/exp/linear_s0/test_orig.npz"] == "test_metrics_orig.json"
    assert list((tmp_path / "mirror" / "logs").glob("merge_rerun_*.json"))
    assert mod.main(args) == 0                                   # a second pass finds everything in place


def test_check_only_writes_nothing(tmp_path):
    mod = _script()
    args, _ = _setup(tmp_path, mod)
    assert mod.main(args + ["--check-only"]) == 0
    assert _arrays_in(tmp_path) == []


def test_a_checkpoint_with_another_hash_stops_everything(tmp_path):
    mod = _script()
    args, _ = _setup(tmp_path, mod, weights=b"head-B")
    assert mod.main(args) == 1
    assert _arrays_in(tmp_path) == []


def test_predictions_that_do_not_give_the_recorded_metrics_stop_everything(tmp_path):
    mod = _script()

    def tamper(summary, report, grid_summary):
        report["top1"] += 0.125                                   # one test image more right than the arrays say
    args, _ = _setup(tmp_path, mod, tamper=tamper)
    assert mod.main(args) == 1
    assert _arrays_in(tmp_path) == []


def test_an_unhashed_grid_checkpoint_needs_its_validation_check(tmp_path):
    mod = _script()

    def tamper(summary, report, grid_summary):
        grid_summary["val_top1"] += 0.125
    args, _ = _setup(tmp_path, mod, tamper=tamper)
    assert mod.main(args) == 1
    assert _arrays_in(tmp_path) == []


def test_a_different_file_already_in_place_is_never_overwritten(tmp_path):
    mod = _script()
    args, _ = _setup(tmp_path, mod)
    other = tmp_path / "mirror" / "experiments" / "exp" / "linear_s0" / "best.pt"
    other.write_bytes(b"something else")
    assert mod.main(args) == 1
    assert other.read_bytes() == b"something else"
    assert _arrays_in(tmp_path) == ["mirror/experiments/exp/linear_s0/best.pt"]
