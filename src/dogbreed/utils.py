"""Small shared helpers: seeding, JSON, logging, environment record, code fingerprint."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PKG_DIR = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_jsonable(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent, default=_to_jsonable), encoding="utf-8")
    os.replace(tmp, path)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_logger(name: str = "dogbreed", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(Path(log_file).resolve())
                   for h in logger.handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(fh)
    return logger


def code_hash() -> str:
    """SHA-1 over every .py file of the package: identifies the code version of a run."""
    h = hashlib.sha1()
    for p in sorted(PKG_DIR.rglob("*.py")):
        h.update(p.relative_to(PKG_DIR).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def env_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "processor": platform.processor(), "torch": torch.__version__,
            "numpy": np.__version__, "cuda_available": torch.cuda.is_available(),
            "cpu_threads": torch.get_num_threads(), "code_hash": code_hash(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        import timm
        info["timm"] = timm.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    return info


def auto_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype(device: torch.device, training: bool = False):
    """Mixed-precision dtype for ``device``, or None for plain fp32.

    bf16 only where it is native (compute capability >= 8: A100, L4, H100). On a T4 bf16 is
    emulated (``torch.cuda.is_bf16_supported()`` still says True), so inference uses fp16 and
    training stays in fp32 -- the heads are small, and fp32 needs no loss scaling.
    """
    if device.type != "cuda":
        return None
    if torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    return None if training else torch.float16


def autocast(device: torch.device, training: bool = False):
    from contextlib import nullcontext
    dt = amp_dtype(device, training)
    return torch.autocast("cuda", dtype=dt) if dt is not None else nullcontext()


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def __call__(self) -> float:
        return time.perf_counter() - self.t0
