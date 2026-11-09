"""Image transforms and the mapping of annotation boxes into the network's input frame."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def default_resize(img_size: int) -> int:
    """Shorter side before the centre crop (224 -> 256, the usual 0.875 crop ratio)."""
    return int(round(img_size / 0.875))


def eval_transform(img_size: int = 224, resize: Optional[int] = None,
                   mean: Sequence[float] = IMAGENET_MEAN, std: Sequence[float] = IMAGENET_STD):
    resize = resize or default_resize(img_size)
    return T.Compose([
        T.Resize(resize, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def train_transform(img_size: int = 224, mean: Sequence[float] = IMAGENET_MEAN,
                    std: Sequence[float] = IMAGENET_STD, scale: Tuple[float, float] = (0.35, 1.0)):
    """Augmentation for end-to-end fine-tuning (cached-feature training uses eval views)."""
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=scale, interpolation=InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def resized_size(w: int, h: int, resize: int) -> Tuple[int, int]:
    """torchvision Resize(int): shorter side -> resize, longer side floored."""
    if w <= h:
        return resize, int(resize * h / w)
    return int(resize * w / h), resize


def box_to_input_frame(box: Sequence[float], w: int, h: int, img_size: int = 224,
                       resize: Optional[int] = None, flipped: bool = False) -> Optional[np.ndarray]:
    """Map a box (x1, y1, x2, y2) in original pixels to the centre-cropped input frame.

    Returns None if the box is missing or falls entirely outside the crop.
    """
    if box is None or any(v is None or (isinstance(v, float) and np.isnan(v)) for v in box):
        return None
    resize = resize or default_resize(img_size)
    nw, nh = resized_size(w, h, resize)
    sx, sy = nw / w, nh / h
    left = int(round((nw - img_size) / 2.0))
    top = int(round((nh - img_size) / 2.0))
    x1, y1, x2, y2 = (box[0] * sx - left, box[1] * sy - top, box[2] * sx - left, box[3] * sy - top)
    if flipped:
        x1, x2 = img_size - x2, img_size - x1
    b = np.clip(np.array([x1, y1, x2, y2], dtype=np.float64), 0, img_size)
    if b[2] - b[0] < 1 or b[3] - b[1] < 1:
        return None
    return b


def box_token_coverage(box: Optional[np.ndarray], grid: int, img_size: int = 224) -> Optional[np.ndarray]:
    """Fraction of each of the grid x grid token cells covered by the box -> (grid*grid,)."""
    if box is None:
        return None
    edges = np.linspace(0, img_size, grid + 1)
    lo, hi = edges[:-1], edges[1:]
    cov_x = np.clip(np.minimum(hi, box[2]) - np.maximum(lo, box[0]), 0, None) / (hi - lo)
    cov_y = np.clip(np.minimum(hi, box[3]) - np.maximum(lo, box[1]), 0, None) / (hi - lo)
    return np.outer(cov_y, cov_x).reshape(-1)
