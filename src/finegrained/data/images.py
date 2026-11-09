"""Image access for Study 2: Study 1's ImageSource plus the per-dataset crop (Aircraft banner)."""
from __future__ import annotations

from PIL import Image

from dogbreed.datasets.sources import ImageSource


class CroppedSource:
    """Wraps an ImageSource; ``open`` removes ``bottom`` pixels from every image (0 = no crop)."""

    def __init__(self, source: ImageSource, bottom: int = 0):
        self.source, self.bottom = source, int(bottom)

    def read_bytes(self, key: str) -> bytes:
        return self.source.read_bytes(key)

    def __contains__(self, key: str) -> bool:
        return key in self.source

    def open(self, key: str) -> Image.Image:
        im = self.source.open(key)
        if self.bottom:
            w, h = im.size
            if h <= self.bottom:
                raise ValueError(f"{key}: {h} px tall, cannot remove a {self.bottom} px banner")
            im = im.crop((0, 0, w, h - self.bottom))
        return im
