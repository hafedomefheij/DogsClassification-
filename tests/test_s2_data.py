"""Study 2 data stage on tiny synthetic copies of the CUB and Aircraft folder layouts."""
import hashlib
import shutil

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dogbreed.datasets.sources import ImageSource
from finegrained.data import audit
from finegrained.data.download import check
from finegrained.data.images import CroppedSource
from finegrained.data.manifests import aircraft_raw, cub_raw, finalise, manifest_sha256, split_counts
from finegrained.data.specs import DATASETS, Archive


def _img(path, seed, size=(64, 48)):
    rng = np.random.default_rng(seed)
    a = (rng.random((size[1] // 8, size[0] // 8, 3)) * 255).astype(np.uint8)
    Image.fromarray(a).resize(size, Image.NEAREST).save(path, quality=95)


@pytest.fixture()
def fake_cub(tmp_path):
    base = tmp_path / "x" / "CUB_200_2011"
    (base / "images").mkdir(parents=True)
    classes, images, labels, split, boxes = [], [], [], [], []
    iid = 0
    for c in range(1, 201):
        name = f"{c:03d}.Bird_{c}"
        classes.append(f"{c} {name}")
        (base / "images" / name).mkdir()
        for k in range(6):
            iid += 1
            rel = f"{name}/Bird_{c}_{k}.jpg"
            _img(base / "images" / rel, seed=iid)
            images.append(f"{iid} {rel}")
            labels.append(f"{iid} {c}")
            split.append(f"{iid} {1 if k < 4 else 0}")
            boxes.append(f"{iid} 1.0 2.0 30.0 20.0")
    # image 2 (train) is a byte copy of image 5 (test): an exact cross-split duplicate
    shutil.copy(base / "images" / images[4].split()[1], base / "images" / images[1].split()[1])
    for fname, lines in (("classes.txt", classes), ("images.txt", images), ("image_class_labels.txt", labels),
                         ("train_test_split.txt", split), ("bounding_boxes.txt", boxes)):
        (base / fname).write_text("\n".join(lines) + "\n")
    return tmp_path


@pytest.fixture()
def fake_aircraft(tmp_path):
    data = tmp_path / "x" / "fgvc-aircraft-2013b" / "data"
    (data / "images").mkdir(parents=True)
    variants = ["707-320", "A300B4", "Cessna 172"]
    (data / "variants.txt").write_text("\n".join(variants) + "\n")
    n = 0
    for split in ("train", "val", "test"):
        var, fam, man = [], [], []
        for v in variants:
            for k in range(2):
                n += 1
                iid = f"{1000000 + n:07d}"
                _img(data / "images" / f"{iid}.jpg", seed=n, size=(80, 60))
                var.append(f"{iid} {v}")
                fam.append(f"{iid} Fam {v[:2]}")
                man.append(f"{iid} Maker {v[0]}")
        for level, lines in (("variant", var), ("family", fam), ("manufacturer", man)):
            (data / f"images_{level}_{split}.txt").write_text("\n".join(lines) + "\n")
    return tmp_path


def test_cub_manifest_and_split(fake_cub):
    df, root = cub_raw(fake_cub / "x")
    assert len(df) == 1200 and df["label"].nunique() == 200
    assert set(df["official_split"]) == {"train", "test"}
    assert df["key"].iloc[0] == "001.Bird_1/Bird_1_0.jpg" and (root / df["key"].iloc[0]).exists()
    fin = finalise(df, val_from_train=0.15, seed=0, exclude_ids=["cub_00002"])
    counts = split_counts(fin)
    assert counts["test"] == 400 and counts["excluded"] == 1
    assert counts["val"] == 200          # max(1, round(0.15 * 4)) = 1 per class
    assert fin.loc[fin["id"] == "cub_00002", "split"].item() == "excluded"
    again = finalise(df, val_from_train=0.15, seed=0, exclude_ids=["cub_00002"])
    assert manifest_sha256(fin) == manifest_sha256(again)
    assert manifest_sha256(finalise(df, 0.15, seed=1, exclude_ids=["cub_00002"])) != manifest_sha256(fin)


def test_aircraft_manifest_crop_and_labels(fake_aircraft):
    df, root = aircraft_raw(fake_aircraft / "x")
    assert len(df) == 18 and df["label"].nunique() == 3
    assert df.loc[df["class_name"] == "Cessna 172", "label"].nunique() == 1    # names with spaces
    assert (df["split"] == df["official_split"]).all()
    src = CroppedSource(ImageSource(root), bottom=20)
    assert src.open(df["key"].iloc[0]).size == (80, 40)
    assert ImageSource(root).open(df["key"].iloc[0]).size == (80, 60)
    fin = finalise(df, None, 0, exclude_ids=[df["id"].iloc[0], df.loc[df["split"] == "test", "id"].iloc[0]])
    assert split_counts(fin) == {"excluded": 1, "test": 6, "train": 5, "val": 6}   # test is never excluded


def test_audit_finds_the_planted_copy(fake_cub):
    df, root = cub_raw(fake_cub / "x")
    h = audit.hash_table(df, CroppedSource(ImageSource(root)), workers=4)
    assert h["ok"].all()
    pairs = audit.candidate_pairs(df, h, max_phash=8)
    hit = pairs[(pairs["id_a"] == "cub_00002") & (pairs["id_b"] == "cub_00005")]
    assert len(hit) == 1 and hit["phash_d"].item() == 0 and hit["same_file"].item()
    rep = audit.summarise(df, h, pairs)
    assert rep["exact_file"]["groups"] == 1 and rep["exact_file"]["groups_by_splits"] == {"train+test": 1}
    assert rep["exact_file"]["groups_with_label_conflict"] == 0
    row = [r for r in rep["phash_candidates"] if r["phash_le"] == 0 and r["splits"] == "train-test"][0]
    assert row["pairs"] >= 1


def test_label_conflict_is_reported(fake_cub):
    df, root = cub_raw(fake_cub / "x")
    df.loc[df["id"] == "cub_00005", "label"] = 7          # the copy now carries another label
    h = audit.hash_table(df, CroppedSource(ImageSource(root)), workers=2)
    rep = audit.summarise(df, h, audit.candidate_pairs(df, h))
    assert rep["exact_file"]["groups_with_label_conflict"] == 1


def test_checksum_check(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"fine-grained")
    sha, md5 = hashlib.sha256(b"fine-grained").hexdigest(), hashlib.md5(b"fine-grained").hexdigest()
    assert check(p, Archive("a.bin", "u", sha256=sha, md5=md5, size=12))["verified"]
    bad = check(p, Archive("a.bin", "u", sha256=sha, md5="0" * 32))
    assert not bad["verified"] and bad["failed"] == ["md5"]
    assert not check(p, Archive("a.bin", "u"))["verified"]       # no published value: not verified


def test_specs_are_complete():
    for spec in DATASETS.values():
        assert spec.archives and spec.citation and spec.licence
    assert DATASETS["cub"].archives[0].sha256 and DATASETS["aircraft"].archives[0].md5
    assert DATASETS["aircraft"].crop_bottom_px == 20
