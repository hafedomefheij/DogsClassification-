"""Statistics for comparing models on the same test images."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.stats import binomtest, t as student_t


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    out = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}
    if len(v) > 1:
        half = student_t.ppf(0.975, len(v) - 1) * out["std"] / np.sqrt(len(v))
        out["ci95"] = [out["mean"] - half, out["mean"] + half]
    return out


def bootstrap_ci(correct: np.ndarray, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI of an accuracy (resampling test images)."""
    rng = np.random.default_rng(seed)
    c = np.asarray(correct, dtype=np.float64)
    idx = rng.integers(0, len(c), (n_boot, len(c)))
    accs = c[idx].mean(1)
    return float(c.mean()), [float(np.quantile(accs, alpha / 2)), float(np.quantile(accs, 1 - alpha / 2))]


def paired_bootstrap(correct_a: np.ndarray, correct_b: np.ndarray, n_boot: int = 10000,
                     seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    """Accuracy difference A - B on the same images with a paired percentile bootstrap."""
    a = np.asarray(correct_a, dtype=np.float64)
    b = np.asarray(correct_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired comparison needs the same test images")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    d = (a[idx] - b[idx]).mean(1)
    return {"diff": float((a - b).mean()), "ci95": [float(np.quantile(d, alpha / 2)),
                                                     float(np.quantile(d, 1 - alpha / 2))],
            "p_boot_le0": float((d <= 0).mean())}


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> Dict[str, float]:
    """Exact McNemar test on discordant pairs (A right & B wrong vs A wrong & B right)."""
    a = np.asarray(correct_a, bool)
    b = np.asarray(correct_b, bool)
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return {"a_only": n10, "b_only": n01, "p_value": float(p)}
