"""Rival-conditioned evidence attention (RCEA): the central mechanism of this study.

Setting
-------
For one image the backbone gives spatial tokens X = [x_1 .. x_N] (x_i in R^D) and a global
descriptor g. A global linear head gives logits z0 in R^C. The candidate set
S = top-K(z0) holds the classes the global head is hesitating between (the image's own
"confusion set").

Mechanism
---------
Every class c owns a learned embedding e_c in R^d. For each candidate c in S, RCEA forms a
*contrastive* query against the other candidates (its rivals)

    q_c = e_c - sum_{c' in S, c' != c} w_{c,c'} e_{c'}                       (1)

with rival weights w_{c,.} either uniform, 1/(K-1), or a softmax of the (detached) global
logits of the rivals, so that the most threatening rival dominates. The query collects the
local evidence that favours c over its current rivals,

    a_{c,i} = softmax_i( (q_c + b)^T k_i / (sqrt(d_h) * tau) ),   k_i = W_k LN(x_i)   (2)
    r_c     = sum_i a_{c,i} v_i,                                   v_i = W_v LN(x_i)   (3)

and scores it along the same direction,

    s_c = <q_c, r_c> / sqrt(d_h)        (summed over H heads of size d_h = d / H)     (4)

The final logits are z_c = z0_c + gamma * s_c for c in S, and z_c = z0_c otherwise.
b is a shared, class-agnostic attention bias ("look at the dog"), gamma a learned scale.

Why the difference matters (the hypothesis this code tests)
-----------------------------------------------------------
Write e_c = m + delta_c, with m the part all breeds share. A class-specific query
(q_c = e_c, as in CSRA or ML-Decoder) is dominated by m and keeps attending to the generic
"dog" patches every breed has. Equation (1) cancels m exactly and keeps only what separates
c from the breeds it is actually being confused with *in this image*. With K = 2 the two
queries are exact negatives (q_a = -q_b = e_a - e_b): each candidate collects the patches
lying furthest toward its own side of the pairwise discriminant direction, a soft-max
pooling of local evidence instead of the global head's mean pooling.

Controls (same parameters, only the query changes; see ``QUERY_MODES``)
-----------------------------------------------------------------------
* ``class``      q_c = e_c                               class-specific attention
* ``center``     q_c = e_c - mean over ALL classes        static contrast (centring only)
* ``random``     q_c = e_c - mean of K-1 random classes   contrast, but not the confusion set
* ``rival``      equation (1), uniform weights            proposed
* ``rival_soft`` equation (1), softmax rival weights      proposed variant
* ``pairwise``   one query per (candidate, rival) pair, aggregated by a soft-min  (below)

If ``rival`` beats ``center`` and ``random`` at equal parameters, the gain comes from
conditioning on the image's own confusion set, not from centring or from contrast alone.

A caveat that shapes the comparison: with uniform weights, equation (1) equals
(K / (K - 1)) (e_c - mean_{c' in S} e_{c'}), i.e. centring on the *shortlist* mean (the factor
K/(K-1) is absorbed by the learned embeddings). ``rival`` is therefore "shortlist centring",
and the controls separate no centring (``class``), global centring (``center``), centring on
a random shortlist (``random``) and centring on the image's own shortlist (``rival``).
``pairwise`` is the variant that is NOT reducible to centring:

    q_{c,r} = e_c - e_r                       for every rival r in S, r != c        (1')
    s_{c,r} = equations (2)-(4) with q_{c,r}   (its own attention map per pair)
    s_c     = softmin_t over r of s_{c,r} = -t log mean_r exp(-s_{c,r} / t)         (4')

so each candidate is scored by its evidence against its *strongest* rival (t -> 0: the
minimum; t -> inf: the mean over rivals). It costs K(K-1) queries instead of K (20 for
K = 5, still far below one query per class) and has exactly the same parameters.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

QUERY_MODES = ("class", "center", "random", "rival", "rival_soft", "pairwise")


def select_candidates(z0: torch.Tensor, k: int, y: Optional[torch.Tensor] = None,
                      gt_insert: bool = False) -> torch.Tensor:
    """Top-k class indices of ``z0`` (B, C) -> (B, k), selected without gradient.

    With ``gt_insert`` (training only) the true class replaces the weakest candidate when it
    is missing, so the re-ranker always learns to separate the true class from its hardest
    rivals. Indices stay unique per row.
    """
    cand = z0.detach().topk(k, dim=1).indices
    if gt_insert and y is not None:
        present = (cand == y[:, None]).any(dim=1)
        if (~present).any():
            cand = cand.clone()
            cand[~present, -1] = y[~present]
    return cand


def rival_weights(z_cand: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Softmax weights over the rivals of every candidate.

    z_cand: (B, K) detached global logits of the candidates.
    Returns w: (B, K, K) with w[b, c, c] = 0 and rows summing to 1 over c' != c.
    """
    b, k = z_cand.shape
    logits = (z_cand / temperature)[:, None, :].expand(b, k, k)
    eye = torch.eye(k, dtype=torch.bool, device=z_cand.device)
    logits = logits.masked_fill(eye, float("-inf"))
    return logits.softmax(dim=-1)


def contrastive_queries(class_emb: torch.Tensor, cand: torch.Tensor, mode: str,
                        z0: Optional[torch.Tensor] = None, rival_temp: float = 1.0,
                        generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Build the query of every candidate.

    class_emb: (C, d) learned class embeddings; cand: (B, K) candidate indices.
    Returns q: (B, K, d).
    """
    if mode not in QUERY_MODES:
        raise ValueError(f"unknown query mode {mode!r}; choose from {QUERY_MODES}")
    if mode == "pairwise":
        raise ValueError("pairwise queries are (B, K, K-1, d): use pairwise_queries()")
    e = class_emb[cand]                                   # (B, K, d)
    b, k, _ = e.shape
    if mode == "class":
        return e
    if mode == "center":
        return e - class_emb.mean(dim=0)
    if k < 2:
        raise ValueError(f"mode {mode!r} needs k >= 2 candidates")
    if mode == "rival":
        total = e.sum(dim=1, keepdim=True)
        return e - (total - e) / (k - 1)
    if mode == "rival_soft":
        if z0 is None:
            raise ValueError("rival_soft needs the global logits z0")
        w = rival_weights(z0.detach().gather(1, cand), rival_temp)   # (B, K, K)
        return e - torch.bmm(w, e)
    # mode == "random": K-1 random classes per candidate, never the candidate itself.
    c = class_emb.shape[0]
    r = torch.randint(0, c - 1, (b, k, k - 1), device=cand.device, generator=generator)
    r = r + (r >= cand[..., None]).long()                 # skip the candidate's own index
    return e - class_emb[r].mean(dim=2)


def rival_index(k: int, device=None) -> torch.Tensor:
    """(K, K-1) table: row c lists the candidate positions r != c, in order."""
    idx = torch.arange(k, device=device).expand(k, k)
    return idx[~torch.eye(k, dtype=torch.bool, device=device)].view(k, k - 1)


def pairwise_queries(class_emb: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
    """Equation (1'): q[b, c, j] = e_{cand[b, c]} - e_{cand[b, rival_index[c, j]]}.

    class_emb: (C, d); cand: (B, K) with K >= 2. Returns (B, K, K-1, d).
    """
    k = cand.shape[1]
    if k < 2:
        raise ValueError("pairwise queries need k >= 2 candidates")
    e = class_emb[cand]                                   # (B, K, d)
    return e[:, :, None, :] - e[:, rival_index(k, cand.device), :]


def soft_min(s: torch.Tensor, temperature: float) -> torch.Tensor:
    """Equation (4'): -t log mean exp(-s / t) over the last dim (t <= 0: the exact minimum).

    Normalised by the mean, so equal inputs return that same value for every t.
    """
    if temperature <= 0:
        return s.min(dim=-1).values
    n = s.shape[-1]
    return -temperature * (torch.logsumexp(-s / temperature, dim=-1) - math.log(n))


class RivalEvidenceAttention(nn.Module):
    """Equations (2)-(4): pool evidence with candidate queries and score it.

    Args:
        dim_token: D, width of the backbone tokens.
        dim: d, width of the query/key/value space (split into ``heads``).
        heads: H, number of evidence heads.
        attn_temp: tau in equation (2).
        q_bias: learn the shared class-agnostic attention bias b.
    """

    def __init__(self, dim_token: int, dim: int = 256, heads: int = 4, attn_temp: float = 1.0,
                 q_bias: bool = True):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim, self.heads, self.dh = dim, heads, dim // heads
        self.attn_temp = attn_temp
        self.norm = nn.LayerNorm(dim_token)
        self.to_k = nn.Linear(dim_token, dim, bias=False)
        self.to_v = nn.Linear(dim_token, dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim)) if q_bias else None
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

    def keys_values(self, tokens: torch.Tensor):
        x = self.norm(tokens)                                  # (B, N, D)
        bsz, n, _ = x.shape
        k = self.to_k(x).view(bsz, n, self.heads, self.dh).transpose(1, 2)   # (B, H, N, dh)
        v = self.to_v(x).view(bsz, n, self.heads, self.dh).transpose(1, 2)
        return k, v

    def forward(self, q: torch.Tensor, tokens: torch.Tensor,
                token_mask: Optional[torch.Tensor] = None, return_attn: bool = False):
        """q: (B, Q, d) queries; tokens: (B, N, D); token_mask: (B, N) True = keep.

        Returns scores s: (B, Q) and, if asked, attention a: (B, H, Q, N).
        """
        k, v = self.keys_values(tokens)
        bsz, nq, _ = q.shape
        qa = q if self.q_bias is None else q + self.q_bias
        qa = qa.view(bsz, nq, self.heads, self.dh).transpose(1, 2)           # (B, H, Q, dh)
        logits = qa @ k.transpose(-1, -2) / (math.sqrt(self.dh) * self.attn_temp)
        if token_mask is not None:
            logits = logits.masked_fill(~token_mask[:, None, None, :], float("-inf"))
        a = logits.softmax(dim=-1)                                           # (B, H, Q, N)
        r = a @ v                                                            # (B, H, Q, dh)
        qs = q.view(bsz, nq, self.heads, self.dh).transpose(1, 2)
        s = (qs * r).sum(dim=-1).sum(dim=1) / math.sqrt(self.dh)             # (B, Q)
        return (s, a) if return_attn else (s, None)


class CandidateReRanker(nn.Module):
    """Equations (1)-(4) plus the logit update: the full RCEA re-ranking stage.

    forward(z0, tokens, y) -> dict(logits, cand, scores[, attn])
    """

    def __init__(self, num_classes: int, dim_token: int, dim: int = 256, heads: int = 4,
                 k: int = 5, mode: str = "rival", rival_temp: float = 1.0,
                 attn_temp: float = 1.0, gt_insert: bool = True, q_bias: bool = True,
                 gamma_init: float = 1.0, emb_std: float = 0.02, pair_temp: float = 1.0):
        super().__init__()
        if mode not in QUERY_MODES:
            raise ValueError(f"unknown query mode {mode!r}")
        if k > num_classes:
            raise ValueError("k cannot exceed the number of classes")
        if mode == "pairwise" and k < 2:
            raise ValueError("pairwise mode needs k >= 2")
        self.num_classes, self.k, self.mode = num_classes, k, mode
        self.rival_temp, self.gt_insert, self.pair_temp = rival_temp, gt_insert, pair_temp
        self.class_emb = nn.Parameter(torch.randn(num_classes, dim) * emb_std)
        self.attn = RivalEvidenceAttention(dim_token, dim, heads, attn_temp, q_bias)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, z0: torch.Tensor, tokens: torch.Tensor, y: Optional[torch.Tensor] = None,
                token_mask: Optional[torch.Tensor] = None, return_attn: bool = False,
                cand: Optional[torch.Tensor] = None):
        if cand is None:
            cand = select_candidates(z0, self.k, y, self.gt_insert and self.training)
        out = {}
        if self.mode == "pairwise":
            bsz, k = cand.shape
            q = pairwise_queries(self.class_emb, cand)                       # (B, K, K-1, d)
            s_pair, a = self.attn(q.reshape(bsz, k * (k - 1), -1), tokens, token_mask,
                                  return_attn)
            s_pair = s_pair.view(bsz, k, k - 1)
            s = soft_min(s_pair, self.pair_temp)
            out["pair_scores"] = s_pair
            if a is not None:                                                # (B, H, K, K-1, N)
                a = a.view(bsz, a.shape[1], k, k - 1, a.shape[-1])
        else:
            q = contrastive_queries(self.class_emb, cand, self.mode, z0, self.rival_temp)
            s, a = self.attn(q, tokens, token_mask, return_attn)
        logits = z0.scatter_add(1, cand, self.gamma * s)
        out.update({"logits": logits, "cand": cand, "scores": s, "queries": q})
        if return_attn:
            out["attn"] = a
        return out


class AllClassEvidence(nn.Module):
    """Class-specific attention over ALL classes (CSRA / ML-Decoder style baseline).

    Every class c attends with q_c = e_c and adds gamma * <e_c, r_c> to its logit.
    Cost grows with C instead of K.
    """

    def __init__(self, num_classes: int, dim_token: int, dim: int = 256, heads: int = 4,
                 attn_temp: float = 1.0, q_bias: bool = True, gamma_init: float = 1.0,
                 emb_std: float = 0.02):
        super().__init__()
        self.class_emb = nn.Parameter(torch.randn(num_classes, dim) * emb_std)
        self.attn = RivalEvidenceAttention(dim_token, dim, heads, attn_temp, q_bias)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, z0, tokens, y=None, token_mask=None, return_attn=False):
        q = self.class_emb[None].expand(tokens.shape[0], -1, -1)
        s, a = self.attn(q, tokens, token_mask, return_attn)
        out = {"logits": z0 + self.gamma * s, "scores": s}
        if return_attn:
            out["attn"] = a
        return out


def random_token_mask(bsz: int, n: int, drop: float, device=None,
                      generator: Optional[torch.Generator] = None) -> Optional[torch.Tensor]:
    """Boolean keep-mask that drops a fraction ``drop`` of tokens per sample (>= 1 kept)."""
    if drop <= 0:
        return None
    keep = torch.rand(bsz, n, device=device, generator=generator) >= drop
    empty = ~keep.any(dim=1)
    if empty.any():
        keep[empty, 0] = True
    return keep


def pairwise_margin(logits: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """z_a - z_b per row; a, b are (B,) class indices."""
    return logits.gather(1, a[:, None]).squeeze(1) - logits.gather(1, b[:, None]).squeeze(1)


__all__ = [
    "QUERY_MODES", "select_candidates", "rival_weights", "contrastive_queries",
    "rival_index", "pairwise_queries", "soft_min",
    "RivalEvidenceAttention", "CandidateReRanker", "AllClassEvidence", "random_token_mask",
    "pairwise_margin", "F",
]
