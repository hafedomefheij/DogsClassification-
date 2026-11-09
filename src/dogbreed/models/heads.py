"""Classification heads that read cached backbone outputs (g, tokens).

Every head returns a dict with ``logits`` (B, C), the prediction used for every metric, and,
for two-stage heads, ``logits_global`` (the stage-1 linear head) and ``cand`` (B, K).

Heads compared in this study (same optimiser, epochs and features for all of them):

  name             what it is                                                 role
  ---------------  ---------------------------------------------------------  ---------------
  linear           LayerNorm + linear on g (DINOv2 linear evaluation)         baseline B0
  attnprobe        M learned class-agnostic queries pool the tokens;          baseline B1
                   linear on [g ; pooled]  (attentive probe)
  classattn_all    class-specific attention for ALL C classes (CSRA /         baseline B2
                   ML-Decoder style), z = z0 + gamma <e_c, r_c>
  rerank:<mode>    top-K candidate re-ranking with the RCEA module; <mode> in  proposed + controls
                   {rival, rival_soft} (proposed) or {class, center, random}
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from ..modules.rival_attention import QUERY_MODES, AllClassEvidence, CandidateReRanker


class GlobalLinear(nn.Module):
    def __init__(self, dim_global: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim_global)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(dim_global, num_classes)
        nn.init.zeros_(self.fc.bias)
        nn.init.normal_(self.fc.weight, std=0.01)

    def forward(self, g):
        return self.fc(self.drop(self.norm(g)))


class LinearProbe(nn.Module):
    def __init__(self, dim_global, dim_token, num_classes, dropout=0.0, **_):
        super().__init__()
        self.global_head = GlobalLinear(dim_global, num_classes, dropout)

    def forward(self, g, tokens=None, y=None, token_mask=None, return_attn=False):
        z = self.global_head(g)
        return {"logits": z, "logits_global": z}


class AttentiveProbe(nn.Module):
    """Class-agnostic attentive pooling: M learned queries, multi-head cross-attention."""

    def __init__(self, dim_global, dim_token, num_classes, dim=256, heads=4, queries=4,
                 dropout=0.0, **_):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads, self.dh, self.m = heads, dim // heads, queries
        self.norm = nn.LayerNorm(dim_token)
        self.to_k = nn.Linear(dim_token, dim, bias=False)
        self.to_v = nn.Linear(dim_token, dim, bias=False)
        self.queries = nn.Parameter(torch.randn(queries, dim) * 0.02)
        self.norm_g = nn.LayerNorm(dim_global)
        self.norm_p = nn.LayerNorm(queries * dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(dim_global + queries * dim, num_classes)
        nn.init.zeros_(self.fc.bias)
        nn.init.normal_(self.fc.weight, std=0.01)

    def forward(self, g, tokens, y=None, token_mask=None, return_attn=False):
        b, n, _ = tokens.shape
        x = self.norm(tokens)
        k = self.to_k(x).view(b, n, self.heads, self.dh).transpose(1, 2)
        v = self.to_v(x).view(b, n, self.heads, self.dh).transpose(1, 2)
        q = self.queries.view(1, self.m, self.heads, self.dh).transpose(1, 2)   # (1, H, M, dh)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dh)                   # (B, H, M, N)
        if token_mask is not None:
            logits = logits.masked_fill(~token_mask[:, None, None, :], float("-inf"))
        a = logits.softmax(-1)
        pooled = (a @ v).transpose(1, 2).reshape(b, -1)                          # (B, M*dim)
        feats = torch.cat([self.norm_g(g), self.norm_p(pooled)], dim=1)
        z = self.fc(self.drop(feats))
        out = {"logits": z}
        if return_attn:
            out["attn"] = a
        return out


class ClassAttentionAll(nn.Module):
    """Baseline B2: class-specific attention for every class (no candidate selection)."""

    def __init__(self, dim_global, dim_token, num_classes, dim=256, heads=4, dropout=0.0,
                 attn_temp=1.0, **_):
        super().__init__()
        self.global_head = GlobalLinear(dim_global, num_classes, dropout)
        self.evidence = AllClassEvidence(num_classes, dim_token, dim, heads, attn_temp)

    def forward(self, g, tokens, y=None, token_mask=None, return_attn=False):
        z0 = self.global_head(g)
        out = self.evidence(z0, tokens, y, token_mask, return_attn)
        out["logits_global"] = z0
        return out


class ReRankHead(nn.Module):
    """Proposed family: global linear head -> top-K candidates -> RCEA re-ranking."""

    def __init__(self, dim_global, dim_token, num_classes, mode="rival", k=5, dim=256, heads=4,
                 dropout=0.0, rival_temp=1.0, attn_temp=1.0, gt_insert=True, q_bias=True,
                 pair_temp=1.0, **_):
        super().__init__()
        self.global_head = GlobalLinear(dim_global, num_classes, dropout)
        self.rerank = CandidateReRanker(num_classes, dim_token, dim, heads, k, mode, rival_temp,
                                        attn_temp, gt_insert, q_bias, pair_temp=pair_temp)

    def forward(self, g, tokens, y=None, token_mask=None, return_attn=False, cand=None):
        z0 = self.global_head(g)
        out = self.rerank(z0, tokens, y, token_mask, return_attn, cand=cand)
        out["logits_global"] = z0
        return out


def build_head(spec: dict, dim_global: int, dim_token: int, num_classes: int) -> nn.Module:
    """spec: {'name': 'linear'|'attnprobe'|'classattn_all'|'rerank', 'mode': ..., ...}."""
    spec = dict(spec)
    name = spec.pop("name")
    if name == "linear":
        return LinearProbe(dim_global, dim_token, num_classes, **spec)
    if name == "attnprobe":
        return AttentiveProbe(dim_global, dim_token, num_classes, **spec)
    if name == "classattn_all":
        return ClassAttentionAll(dim_global, dim_token, num_classes, **spec)
    if name == "rerank":
        if spec.get("mode", "rival") not in QUERY_MODES:
            raise ValueError(f"unknown mode {spec.get('mode')!r}")
        return ReRankHead(dim_global, dim_token, num_classes, **spec)
    raise ValueError(f"unknown head {name!r}")


def head_label(spec: dict) -> str:
    """Short, stable name for tables: e.g. 'rerank-rival-k5'."""
    name = spec["name"]
    if name == "rerank":
        return f"rerank-{spec.get('mode', 'rival')}-k{spec.get('k', 5)}"
    return name


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
