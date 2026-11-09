"""Evaluation metrics: accuracy, per-class and macro scores, calibration, selective
classification, out-of-distribution detection, pairwise confusion.

All functions take numpy logits (N, C) and integer labels (N,).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = z / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def topk_acc(z: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    k = min(k, z.shape[1])
    top = np.argpartition(-z, k - 1, axis=1)[:, :k]
    return float((top == y[:, None]).any(axis=1).mean())


def nll(z: np.ndarray, y: np.ndarray, t: float = 1.0) -> float:
    return float(F.cross_entropy(torch.from_numpy(z / t).double(), torch.from_numpy(y).long()))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error, equal-width confidence bins (Guo et al. 2017)."""
    conf, pred = p.max(1), p.argmax(1)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 15):
    conf, correct = p.max(1), (p.argmax(1) == y)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        rows.append((float((lo + hi) / 2), float(correct[m].mean()) if m.any() else np.nan,
                     float(conf[m].mean()) if m.any() else np.nan, int(m.sum())))
    return rows


def brier(p: np.ndarray, y: np.ndarray) -> float:
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1
    return float(((p - oh) ** 2).sum(1).mean())


def fit_temperature(z_val: np.ndarray, y_val: np.ndarray) -> float:
    """Temperature scaling: T minimising validation NLL (log-parametrised, LBFGS)."""
    z = torch.from_numpy(z_val).double()
    y = torch.from_numpy(y_val).long()
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def rerank_effect(z: np.ndarray, z0: np.ndarray, y: np.ndarray) -> dict:
    """How the second stage changes the first-stage decision.

    flip: argmax changed; fix: first stage wrong -> right; break: right -> wrong.
    The accuracy change equals fix - break.
    """
    p, p0 = np.asarray(z).argmax(1), np.asarray(z0).argmax(1)
    ok, ok0 = p == y, p0 == y
    return {"flip_rate": float((p != p0).mean()), "fix_rate": float((ok & ~ok0).mean()),
            "break_rate": float((~ok & ok0).mean())}


def risk_coverage(conf: np.ndarray, correct: np.ndarray):
    """Selective classification: risk (error rate) of the k most confident predictions."""
    order = np.argsort(-conf, kind="stable")
    err = 1.0 - correct[order].astype(np.float64)
    k = np.arange(1, len(err) + 1)
    risk = np.cumsum(err) / k
    return k / len(err), risk


def aurc(conf: np.ndarray, correct: np.ndarray) -> Tuple[float, float]:
    """Area under the risk-coverage curve, and excess AURC over an oracle ranking."""
    _, risk = risk_coverage(conf, correct)
    a = float(risk.mean())
    _, risk_opt = risk_coverage(correct.astype(np.float64), correct)
    return a, a - float(risk_opt.mean())


def risk_at_coverage(conf: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    cov, risk = risk_coverage(conf, correct)
    i = min(len(risk) - 1, int(np.ceil(coverage * len(risk))) - 1)
    return float(risk[i])


def confidence_scores(z: np.ndarray) -> Dict[str, np.ndarray]:
    """Scores where larger = more confident / more in-distribution."""
    p = softmax(z)
    srt = np.sort(z, axis=1)
    return {"msp": p.max(1), "maxlogit": z.max(1),
            "energy": np.log(np.exp(z - z.max(1, keepdims=True)).sum(1)) + z.max(1),
            "margin": srt[:, -1] - srt[:, -2]}


def ood_metrics(score_id: np.ndarray, score_ood: np.ndarray) -> Dict[str, float]:
    """AUROC and FPR at 95 % TPR (in-distribution = positive)."""
    labels = np.r_[np.ones(len(score_id)), np.zeros(len(score_ood))]
    scores = np.r_[score_id, score_ood]
    thr = np.quantile(score_id, 0.05)
    return {"auroc": float(roc_auc_score(labels, scores)),
            "fpr95": float((score_ood >= thr).mean())}


def confusion_matrix(y: np.ndarray, pred: np.ndarray, c: int) -> np.ndarray:
    m = np.zeros((c, c), dtype=np.int64)
    np.add.at(m, (y, pred), 1)
    return m


def top_confused_pairs(cm: np.ndarray, n: int = 20) -> List[Tuple[int, int, int]]:
    """Unordered class pairs with the most mutual errors: (a, b, errors a->b + b->a)."""
    sym = cm + cm.T
    np.fill_diagonal(sym, 0)
    iu = np.triu_indices_from(sym, k=1)
    vals = sym[iu]
    order = np.argsort(-vals)[:n]
    return [(int(iu[0][i]), int(iu[1][i]), int(vals[i])) for i in order if vals[i] > 0]


def pair_accuracy(y: np.ndarray, pred: np.ndarray, pairs: List[Tuple[int, int, int]]) -> float:
    """Accuracy restricted to test images whose true class belongs to one of the pairs."""
    cls = {a for a, _, _ in pairs} | {b for _, b, _ in pairs}
    m = np.isin(y, list(cls))
    return float((pred[m] == y[m]).mean()) if m.any() else float("nan")


def full_report(z: np.ndarray, y: np.ndarray, num_classes: int, t: Optional[float] = None,
                class_names: Optional[List[str]] = None) -> Dict:
    pred = z.argmax(1)
    p = softmax(z)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=np.arange(num_classes),
                                                    average="macro", zero_division=0)
    per_class = [float((pred[y == c] == c).mean()) if (y == c).any() else float("nan")
                 for c in range(num_classes)]
    correct = pred == y
    conf = confidence_scores(z)
    a, ea = aurc(conf["msp"], correct)
    rep = {"n": int(len(y)), "top1": topk_acc(z, y, 1), "top5": topk_acc(z, y, 5),
           "macro_precision": float(pr), "macro_recall": float(rc), "macro_f1": float(f1),
           "mean_per_class_acc": float(np.nanmean(per_class)), "nll": nll(z, y),
           "ece": ece(p, y), "brier": brier(p, y), "aurc": a, "eaurc": ea,
           "risk_at_90cov": risk_at_coverage(conf["msp"], correct, 0.9),
           "aurc_margin": aurc(conf["margin"], correct)[0], "per_class_acc": per_class}
    if t is not None:
        pt = softmax(z, t)
        rep.update({"temperature": t, "nll_ts": nll(z, y, t), "ece_ts": ece(pt, y)})
    if class_names:
        rep["worst_classes"] = sorted(
            [(class_names[c], per_class[c]) for c in range(num_classes)], key=lambda x: x[1])[:10]
    return rep
