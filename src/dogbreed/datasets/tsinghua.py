"""Tsinghua Dogs (Zou et al., Computational Visual Media 2020) -> manifest.

Release used: the low-resolution images (low-resolution.zip), Low-Annotations (one PASCAL-VOC
style XML per image with <headbndbox> and <bodybndbox>) and TrainValSplit
(TrainAndValList/train.lst: 65,228 images; validation.lst: 5,200 = 40 per breed).
The official "validation" list is the benchmark's evaluation set: we call it ``test`` and never
use it for model selection. Our own validation split is carved out of train.lst.

Class folders look like ``1043-n000001-Shiba_Dog``: the leading number is the breed's total
image count, ``n000001`` its id. Labels are assigned in id order (n000001 -> 0).
"""
from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .sources import norm_key

CLASS_DIR_RE = re.compile(r"^(\d+)-(n\d+)-(.+)$")


def parse_class_dir(name: str):
    m = CLASS_DIR_RE.match(name)
    if not m:
        raise ValueError(f"unexpected Tsinghua class folder {name!r}")
    return int(m.group(1)), m.group(2), m.group(3)


def read_list(path: str | Path):
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    return [norm_key(x) for x in lines if x.strip()]


def _box(obj: ET.Element, tag: str):
    b = obj.find(tag)
    if b is None:
        return None
    try:
        vals = [float(b.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")]
    except (AttributeError, TypeError, ValueError):
        return None
    return vals


def parse_annotation(xml_text: str) -> Dict[str, Optional[float]]:
    """Width, height, head box and body box of the first object, plus the object count."""
    root = ET.fromstring(xml_text.lstrip("﻿").strip())
    size = root.find("size")
    out: Dict[str, Optional[float]] = {"ann_w": None, "ann_h": None, "n_objects": 0}
    if size is not None:
        try:
            out["ann_w"] = float(size.find("width").text)
            out["ann_h"] = float(size.find("height").text)
        except (AttributeError, TypeError, ValueError):
            pass
    objs = root.findall("object")
    out["n_objects"] = len(objs)
    head = body = None
    for obj in objs:
        head = head or _box(obj, "headbndbox")
        body = body or _box(obj, "bodybndbox")
    for prefix, box in (("head", head), ("body", body)):
        for name, v in zip(("x1", "y1", "x2", "y2"), box or (None,) * 4):
            out[f"{prefix}_{name}"] = v
    return out


def annotation_reader(raw: Path, annotations: Optional[str | Path] = None):
    """(class folder, file name) -> XML text or None.

    Reads straight from low-annotations.zip when it is there (one file instead of 70k small
    ones -- opening each of those is the slow part on Windows); otherwise from the folder
    <raw>/annotations/Low-Annotations or the given ``annotations`` path (folder or zip).
    """
    src = Path(annotations) if annotations else (
        raw / "low-annotations.zip" if (raw / "low-annotations.zip").exists()
        else raw / "annotations" / "Low-Annotations")
    if src.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(src)
        index = {norm_key(n): n for n in zf.namelist() if n.lower().endswith(".xml")}

        def read(cdir: str, fname: str) -> Optional[str]:
            n = index.get(f"{cdir}/{fname}.xml")
            return None if n is None else zf.read(n).decode("utf-8-sig", errors="replace")
        return read

    def read_dir(cdir: str, fname: str) -> Optional[str]:
        p = src / cdir / f"{fname}.xml"
        return p.read_text(encoding="utf-8-sig", errors="replace") if p.exists() else None
    return read_dir


def build_manifest(raw_dir: str | Path, annotations_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """One row per image of the official split lists, with labels and GT boxes."""
    raw = Path(raw_dir)
    split_dir = raw / "split" / "TrainAndValList"
    train = read_list(split_dir / "train.lst")
    test = read_list(split_dir / "validation.lst")
    overlap = set(train) & set(test)
    if overlap:
        raise ValueError(f"{len(overlap)} images appear in both official lists")
    read_xml = annotation_reader(raw, annotations_dir)

    class_dirs = sorted({k.split("/")[0] for k in train + test},
                        key=lambda d: parse_class_dir(d)[1])
    label_of = {d: i for i, d in enumerate(class_dirs)}
    rows = []
    for split, keys in (("train", train), ("test", test)):
        for key in keys:
            cdir, fname = key.split("/")
            count, ckey, cname = parse_class_dir(cdir)
            row = {"key": key, "label": label_of[cdir], "class_key": ckey, "class_name": cname,
                   "class_total": count, "official_split": split}
            xml = read_xml(cdir, fname)
            if xml is not None:
                row.update(parse_annotation(xml))
            row["has_ann"] = xml is not None
            rows.append(row)
    df = pd.DataFrame(rows)
    df.insert(0, "dataset", "tsinghua")
    df.insert(0, "id", range(len(df)))
    return df
