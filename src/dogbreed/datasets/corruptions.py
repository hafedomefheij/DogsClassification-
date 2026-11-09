"""Deterministic test-time perturbations for robustness and annotation-condition studies.

Each perturbation takes a PIL image, the manifest row (for the GT boxes) and an integer seed,
and is applied *before* the evaluation transform. Families:

* image quality: ``blur``, ``noise``, ``jpeg``, ``lowres``
* occlusion:     ``occlude`` (random grey square, 25 % of the area), ``head_occlude`` (GT head box)
* background:    ``bg_gray`` / ``bg_blur`` (everything outside the GT body box)
* annotation-assisted inference (oracle boxes at test time): ``crop_body``, ``crop_head``

The annotation-assisted variants use ground-truth boxes and are reported separately from
annotation-free inference; they are never mixed into the main results.
"""
from __future__ import annotations

import io
from typing import Callable, Dict, Optional

import numpy as np
from PIL import Image, ImageFilter

GREY = (124, 116, 104)  # ImageNet mean in 0..255, so a grey patch is ~zero after normalisation


def _box(row, prefix: str, w: int, h: int):
    try:
        b = [float(row[f"{prefix}_{k}"]) for k in ("x1", "y1", "x2", "y2")]
    except (KeyError, TypeError, ValueError):
        return None
    if any(np.isnan(v) for v in b):
        return None
    # annotation size may differ from the decoded size (low-resolution release): rescale
    aw, ah = row.get("ann_w"), row.get("ann_h")
    if aw and ah and not np.isnan(aw) and not np.isnan(ah) and (aw != w or ah != h):
        sx, sy = w / aw, h / ah
        b = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
    b = [max(0, b[0]), max(0, b[1]), min(w, b[2]), min(h, b[3])]
    if b[2] - b[0] < 2 or b[3] - b[1] < 2:
        return None
    return [int(round(v)) for v in b]


def blur(img, row=None, seed=0, radius=3.0):
    return img.filter(ImageFilter.GaussianBlur(radius))


def noise(img, row=None, seed=0, sigma=25.0):
    rng = np.random.default_rng(seed)
    a = np.asarray(img, dtype=np.float32)
    a = a + rng.normal(0, sigma, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def jpeg(img, row=None, seed=0, quality=10):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def lowres(img, row=None, seed=0, factor=4):
    w, h = img.size
    small = img.resize((max(1, w // factor), max(1, h // factor)), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def occlude(img, row=None, seed=0, area=0.25):
    rng = np.random.default_rng(seed)
    w, h = img.size
    side = int(round(np.sqrt(area * w * h)))
    side = min(side, w, h)
    x = int(rng.integers(0, w - side + 1))
    y = int(rng.integers(0, h - side + 1))
    out = img.copy()
    out.paste(GREY, (x, y, x + side, y + side))
    return out


def head_occlude(img, row=None, seed=0):
    b = _box(row, "head", *img.size) if row is not None else None
    if b is None:
        return img
    out = img.copy()
    out.paste(GREY, tuple(b))
    return out


def _outside_body(img, row, fill_fn):
    b = _box(row, "body", *img.size) if row is not None else None
    if b is None:
        return img
    bg = fill_fn(img)
    bg.paste(img.crop(tuple(b)), (b[0], b[1]))
    return bg


def bg_gray(img, row=None, seed=0):
    return _outside_body(img, row, lambda im: Image.new("RGB", im.size, GREY))


def bg_blur(img, row=None, seed=0):
    return _outside_body(img, row, lambda im: im.filter(ImageFilter.GaussianBlur(12)))


def _crop(img, row, prefix, margin=0.1):
    b = _box(row, prefix, *img.size) if row is not None else None
    if b is None:
        return img
    w, h = img.size
    mx, my = margin * (b[2] - b[0]), margin * (b[3] - b[1])
    return img.crop((max(0, int(b[0] - mx)), max(0, int(b[1] - my)),
                     min(w, int(b[2] + mx)), min(h, int(b[3] + my))))


def crop_body(img, row=None, seed=0):
    return _crop(img, row, "body")


def crop_head(img, row=None, seed=0):
    return _crop(img, row, "head")


CORRUPTIONS: Dict[str, Callable] = {
    "blur": blur, "noise": noise, "jpeg": jpeg, "lowres": lowres, "occlude": occlude,
    "head_occlude": head_occlude, "bg_gray": bg_gray, "bg_blur": bg_blur,
    "crop_body": crop_body, "crop_head": crop_head,
}
ANNOTATION_ASSISTED = {"head_occlude", "bg_gray", "bg_blur", "crop_body", "crop_head"}


def apply(name: Optional[str], img: Image.Image, row=None, seed: int = 0) -> Image.Image:
    if not name or name == "none":
        return img
    if name not in CORRUPTIONS:
        raise ValueError(f"unknown corruption {name!r}; choose from {sorted(CORRUPTIONS)}")
    return CORRUPTIONS[name](img, row, seed)
