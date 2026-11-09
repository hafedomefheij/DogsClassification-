"""Read images by key from a directory or a zip archive, without extracting anything.

A *key* is always ``<class folder>/<file name>`` (the last two path parts), so the same manifest
works whether the images sit in a folder (Colab, after extraction) or inside the original zip
(Windows, where extracting 70k small files is slow).
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Optional

from PIL import Image

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def norm_key(path: str) -> str:
    """'.//1043-n000001-Shiba_Dog/n100001.jpeg' -> '1043-n000001-Shiba_Dog/n100001.jpeg'."""
    parts = [p for p in str(path).replace("\\", "/").lstrip("﻿").strip().split("/")
             if p not in ("", ".")]
    return "/".join(parts[-2:])


class ImageSource:
    """Images addressed by key, read from a folder or from a zip file.

    Picklable (DataLoader workers on Windows use spawn): the open zip handle is dropped when
    pickled and reopened lazily in each process.
    """

    def __init__(self, root: str | os.PathLike):
        self.root = str(root)
        self.kind = "zip" if self.root.lower().endswith(".zip") else "dir"
        if not Path(self.root).exists():
            raise FileNotFoundError(self.root)
        self._handle: Optional[zipfile.ZipFile] = None
        self._pid: Optional[int] = None
        self._index: Optional[Dict[str, str]] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_pid"] = None
        return state

    def _zip(self) -> zipfile.ZipFile:
        if self._handle is None or self._pid != os.getpid():
            self._handle = zipfile.ZipFile(self.root)
            self._pid = os.getpid()
            if self._index is None:
                self._index = {norm_key(n): n for n in self._handle.namelist()
                               if n.lower().endswith(IMAGE_EXT)}
        return self._handle

    def keys(self) -> Iterable[str]:
        if self.kind == "zip":
            self._zip()
            return list(self._index)
        root = Path(self.root)
        return [norm_key(str(p.relative_to(root))) for p in root.rglob("*")
                if p.suffix.lower() in IMAGE_EXT]

    def __contains__(self, key: str) -> bool:
        if self.kind == "zip":
            self._zip()
            return norm_key(key) in self._index
        return (Path(self.root) / key).exists()

    def read_bytes(self, key: str) -> bytes:
        if self.kind == "zip":
            zf = self._zip()
            return zf.read(self._index[norm_key(key)])
        return (Path(self.root) / key).read_bytes()

    def open(self, key: str) -> Image.Image:
        """Decode to RGB (palette/CMYK/greyscale/alpha images are converted)."""
        with Image.open(io.BytesIO(self.read_bytes(key))) as im:
            im.load()
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                return bg
            return im.convert("RGB")
