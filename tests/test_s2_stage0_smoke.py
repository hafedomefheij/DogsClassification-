"""End-to-end smoke test of scripts/s2/stage0.py: a fake CUB archive fetched with curl from a file:// URL."""
import dataclasses
import hashlib
import importlib.util
import json
import shutil
import tarfile
from pathlib import Path

import pytest

from finegrained.data import specs

from test_s2_data import fake_cub  # noqa: F401  (fixture)

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("curl") is None, reason="curl not installed")
def test_stage0_end_to_end(fake_cub, tmp_path, monkeypatch):   # noqa: F811
    tgz = tmp_path / "CUB_200_2011.tgz"
    with tarfile.open(tgz, "w:gz") as tf:
        tf.add(fake_cub / "x" / "CUB_200_2011", arcname="CUB_200_2011")
    sha = hashlib.sha256(tgz.read_bytes()).hexdigest()
    arch = specs.Archive("CUB_200_2011.tgz", tgz.resolve().as_uri(), sha256=sha, checksum_source="test")
    spec = dataclasses.replace(specs.CUB, archives=(arch,))
    monkeypatch.setitem(specs.DATASETS, "cub", spec)
    mod_spec = importlib.util.spec_from_file_location("stage0", REPO / "scripts" / "s2" / "stage0.py")
    stage0 = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(stage0)
    monkeypatch.setattr(stage0, "DATASETS", specs.DATASETS)
    rep = stage0.run("cub", tmp_path / "raw", tmp_path / "out", workers=2)
    out = tmp_path / "out" / "cub"
    assert rep["images"] == 1200 and rep["decodable"] == 1200
    assert rep["exact_file"]["groups_by_splits"] == {"train+test": 1}
    assert json.loads((out / "CUB_200_2011.tgz.json").read_text())["verified"]
    for f in ("manifest_raw.csv", "hashes.csv", "phash_pairs.csv", "audit.json", "montage_to_test.png"):
        assert (out / f).exists(), f
    assert rep["montages"]["cross_split_to_test"] >= 1
