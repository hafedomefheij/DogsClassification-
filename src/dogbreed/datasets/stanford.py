"""Stanford Dogs (Khosla et al., FGVC workshop at CVPR 2011) -> manifest.

Release: images.tar (Images/<wnid>-<Name>/<wnid>_<n>.jpg), annotation.tar
(Annotation/<wnid>-<Name>/<wnid>_<n>, PASCAL-VOC XML without extension, one <object> per dog)
and lists.tar (train_list.mat: 12,000 = 100 per breed; test_list.mat: 8,580).
Stanford Dogs has no head boxes; the largest dog box is stored as the body box.
The images are a subset of ImageNet: ImageNet-pretrained backbones may have seen the test
images (see the reproducibility appendix).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


def _mat_list(path: Path):
    from scipy.io import loadmat  # local import: scipy only needed here
    m = loadmat(str(path))
    files = [str(x[0][0]) for x in m["file_list"]]
    labels = [int(x[0]) - 1 for x in m["labels"]]
    return files, labels


def parse_annotation(xml_text: str):
    root = ET.fromstring(xml_text.strip())
    out = {"ann_w": None, "ann_h": None, "n_objects": 0}
    size = root.find("size")
    if size is not None:
        try:
            out["ann_w"] = float(size.find("width").text)
            out["ann_h"] = float(size.find("height").text)
        except (AttributeError, TypeError, ValueError):
            pass
    best, best_area = None, -1.0
    objs = root.findall("object")
    out["n_objects"] = len(objs)
    for obj in objs:
        b = obj.find("bndbox")
        if b is None:
            continue
        box = [float(b.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")]
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area > best_area:
            best, best_area = box, area
    for name, v in zip(("x1", "y1", "x2", "y2"), best or (None,) * 4):
        out[f"body_{name}"] = v
    for name in ("x1", "y1", "x2", "y2"):
        out[f"head_{name}"] = None
    return out


def build_manifest(raw_dir: str | Path) -> pd.DataFrame:
    raw = Path(raw_dir)
    rows = []
    for split, fname in (("train", "train_list.mat"), ("test", "test_list.mat")):
        files, labels = _mat_list(raw / "lists" / fname)
        for f, lab in zip(files, labels):
            cdir, img = f.split("/")
            wnid, cname = cdir.split("-", 1)
            row = {"key": f, "label": lab, "class_key": wnid, "class_name": cname,
                   "official_split": split}
            ann = raw / "Annotation" / cdir / Path(img).stem
            if ann.exists():
                row.update(parse_annotation(ann.read_text(errors="replace")))
                row["has_ann"] = True
            else:
                row["has_ann"] = False
            rows.append(row)
    df = pd.DataFrame(rows)
    df.insert(0, "dataset", "stanford")
    df.insert(0, "id", range(len(df)))
    return df
