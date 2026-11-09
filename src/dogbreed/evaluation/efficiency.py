"""Parameters, FLOPs, latency, throughput and peak memory.

FLOPs are counted with ``torch.utils.flop_counter.FlopCounterMode`` (PyTorch's operator-level
counter; one multiply-add = 2 FLOPs; matmul/conv/attention ops only, element-wise ops not
counted). Latency is the median of timed forward passes after warm-up on the stated device.
"""
from __future__ import annotations

import time
from typing import Callable, Dict

import numpy as np
import torch


def params(module: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in module.parameters())
    train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"params": int(total), "trainable": int(train)}


def flops(fn: Callable[[], object]) -> int:
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            fn()
    return int(fc.get_total_flops())


def latency(fn: Callable[[], object], device: torch.device, warmup: int = 5, iters: int = 20) -> Dict[str, float]:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    t = np.asarray(times)
    return {"median_ms": float(np.median(t) * 1e3), "p90_ms": float(np.quantile(t, 0.9) * 1e3),
            "warmup": warmup, "iters": iters}


def peak_memory(fn: Callable[[], object], device: torch.device) -> Dict[str, float]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            fn()
        torch.cuda.synchronize()
        return {"peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20}
    import psutil
    proc = psutil.Process()
    before = proc.memory_info().rss
    with torch.no_grad():
        fn()
    return {"rss_delta_mb": (proc.memory_info().rss - before) / 2**20,
            "rss_mb": proc.memory_info().rss / 2**20}
