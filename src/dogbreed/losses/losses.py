"""Training objectives.

``full`` (standard):     L = CE(z, y) + alpha * CE(z0, y)
``candidate`` (confusion-restricted):
                         L = CE_S(z_S, y) + alpha * CE(z0, y)
   where CE_S is the cross-entropy over the candidate set S only: the re-ranker is trained as
   a hard-negative softmax over the image's own confusion set (the true class is inserted
   into S during training). For single-stage heads both reduce to CE(z, y).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def compute_loss(out: Dict[str, torch.Tensor], y: torch.Tensor, kind: str = "full",
                 aux_global: float = 1.0, label_smoothing: float = 0.0):
    z = out["logits"]
    terms = {}
    if kind == "candidate" and "cand" in out:
        cand = out["cand"]
        hit = cand == y[:, None]
        valid = hit.any(dim=1)
        pos = hit.float().argmax(dim=1)
        zc = z.gather(1, cand)
        if valid.any():
            terms["rerank"] = F.cross_entropy(zc[valid], pos[valid], label_smoothing=label_smoothing)
        else:
            terms["rerank"] = z.sum() * 0.0
    elif kind in ("full", "candidate"):
        terms["final"] = F.cross_entropy(z, y, label_smoothing=label_smoothing)
    else:
        raise ValueError(f"unknown loss {kind!r}")
    zg = out.get("logits_global")
    if zg is not None and zg is not z and aux_global > 0:
        terms["global"] = aux_global * F.cross_entropy(zg, y, label_smoothing=label_smoothing)
    loss = sum(terms.values())
    return loss, {k: float(v.detach()) for k, v in terms.items()}
