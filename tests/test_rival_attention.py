"""Unit tests of the proposed mechanism: shapes, the algebra of the queries, invariances,
gradient flow and a finite-difference gradient check."""
import math

import pytest
import torch

from dogbreed.modules.rival_attention import (QUERY_MODES, AllClassEvidence, CandidateReRanker,
                                              RivalEvidenceAttention, contrastive_queries,
                                              pairwise_queries, random_token_mask, rival_index,
                                              rival_weights, select_candidates, soft_min)

B, N, D, C, K, d, H = 6, 49, 32, 20, 5, 16, 4


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.randn(B, C), torch.randn(B, N, D), torch.randint(0, C, (B,))


def test_select_candidates_topk_and_gt_insert(batch):
    z0, _, y = batch
    cand = select_candidates(z0, K)
    assert cand.shape == (B, K)
    assert torch.equal(cand, z0.topk(K, 1).indices)
    y = (cand[:, 0] + 1) % C                      # make sure many rows miss the true class
    y[0] = cand[0, 2]                              # one row already contains it
    ins = select_candidates(z0, K, y, gt_insert=True)
    assert (ins == y[:, None]).any(1).all()
    assert torch.equal(ins[0], cand[0])           # untouched when already present
    for row in ins:                                # still unique indices
        assert len(set(row.tolist())) == K


def test_rival_weights_rows_sum_to_one_and_exclude_self():
    w = rival_weights(torch.randn(3, K))
    assert torch.allclose(w.sum(-1), torch.ones(3, K))
    assert torch.all(torch.diagonal(w, dim1=1, dim2=2) == 0)


def test_rival_query_algebra():
    torch.manual_seed(1)
    emb = torch.randn(C, d)
    cand = torch.tensor([[3, 7]])
    q = contrastive_queries(emb, cand, "rival")
    # K = 2: the two queries are exact negatives, q_a = e_a - e_b
    assert torch.allclose(q[0, 0], emb[3] - emb[7])
    assert torch.allclose(q[0, 1], -q[0, 0])
    # a component shared by every class cancels exactly
    shift = torch.randn(d)
    q2 = contrastive_queries(emb + shift, torch.tensor([[1, 4, 9, 2, 5]]), "rival")
    q1 = contrastive_queries(emb, torch.tensor([[1, 4, 9, 2, 5]]), "rival")
    assert torch.allclose(q1, q2, atol=1e-5)
    # ... but not for class-specific queries
    qc1 = contrastive_queries(emb, cand, "class")
    qc2 = contrastive_queries(emb + shift, cand, "class")
    assert not torch.allclose(qc1, qc2)
    # rival queries of one candidate sum to zero over the candidate set (uniform weights)
    q5 = contrastive_queries(emb, torch.tensor([[1, 4, 9, 2, 5]]), "rival")
    assert torch.allclose(q5.sum(1), torch.zeros(1, d), atol=1e-5)


def test_center_and_random_modes():
    torch.manual_seed(2)
    emb = torch.randn(C, d)
    cand = torch.randint(0, C, (B, K))
    qc = contrastive_queries(emb, cand, "center")
    assert torch.allclose(qc, emb[cand] - emb.mean(0), atol=1e-6)
    g = torch.Generator().manual_seed(0)
    qr = contrastive_queries(emb, cand, "random", generator=g)
    assert qr.shape == (B, K, d)
    zs = torch.randn(B, C)
    qs = contrastive_queries(emb, cand, "rival_soft", zs)
    assert qs.shape == (B, K, d)


def test_rival_soft_with_flat_logits_equals_uniform():
    torch.manual_seed(3)
    emb = torch.randn(C, d)
    cand = torch.randint(0, C, (B, K))
    for i in range(B):
        cand[i] = torch.randperm(C)[:K]
    z0 = torch.zeros(B, C)
    assert torch.allclose(contrastive_queries(emb, cand, "rival_soft", z0),
                          contrastive_queries(emb, cand, "rival"), atol=1e-6)


def test_attention_shapes_and_token_permutation_invariance(batch):
    _, tok, _ = batch
    att = RivalEvidenceAttention(D, d, H).eval()
    q = torch.randn(B, K, d)
    s, a = att(q, tok, return_attn=True)
    assert s.shape == (B, K) and a.shape == (B, H, K, N)
    assert torch.allclose(a.sum(-1), torch.ones(B, H, K), atol=1e-5)
    perm = torch.randperm(N)
    s2, _ = att(q, tok[:, perm])
    assert torch.allclose(s, s2, atol=1e-5)       # a set operation over tokens


def test_token_mask_removes_tokens(batch):
    _, tok, _ = batch
    att = RivalEvidenceAttention(D, d, H).eval()
    q = torch.randn(B, K, d)
    keep = torch.ones(B, N, dtype=torch.bool)
    keep[:, 10:] = False
    s_mask, a = att(q, tok, keep, return_attn=True)
    s_cut, _ = att(q, tok[:, :10])
    assert torch.allclose(s_mask, s_cut, atol=1e-5)
    assert torch.all(a[..., 10:] == 0)
    m = random_token_mask(B, N, 0.99)
    assert m.any(1).all()


@pytest.mark.parametrize("mode", QUERY_MODES)
def test_reranker_forward_backward_all_modes(batch, mode):
    z0, tok, y = batch
    z0 = z0.clone().requires_grad_(True)
    rr = CandidateReRanker(C, D, d, H, K, mode).train()
    out = rr(z0, tok, y, return_attn=True)
    assert out["logits"].shape == (B, C) and out["cand"].shape == (B, K)
    # non-candidate logits are untouched
    mask = torch.ones(B, C, dtype=torch.bool)
    mask.scatter_(1, out["cand"], False)
    assert torch.allclose(out["logits"][mask], z0[mask])
    out["logits"].logsumexp(1).mean().backward()
    for name, p in rr.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        if name in ("class_emb", "attn.to_k.weight", "attn.to_v.weight", "gamma"):
            assert p.grad.abs().sum() > 0, f"no gradient reaches {name}"


def test_all_modes_have_identical_parameter_counts():
    counts = {m: sum(p.numel() for p in CandidateReRanker(C, D, d, H, K, m).parameters())
              for m in QUERY_MODES}
    assert len(set(counts.values())) == 1, counts


def test_class_mode_ignores_rivals_rival_mode_does_not(batch):
    _, tok, _ = batch
    torch.manual_seed(4)
    for mode, should_change in (("class", False), ("rival", True)):
        rr = CandidateReRanker(C, D, d, H, 3, mode, gt_insert=False).eval()
        z0 = torch.zeros(B, C)
        c1 = torch.tensor([[0, 1, 2]] * B)
        c2 = torch.tensor([[0, 5, 6]] * B)                 # same class 0, different rivals
        s1 = rr(z0, tok, cand=c1)["scores"][:, 0]
        s2 = rr(z0, tok, cand=c2)["scores"][:, 0]
        assert (not torch.allclose(s1, s2, atol=1e-6)) == should_change, mode


def test_reranker_finite_difference_gradcheck():
    torch.manual_seed(5)
    rr = CandidateReRanker(7, 6, 8, 2, 3, "rival_soft", gt_insert=False).double().eval()
    tok = torch.randn(2, 5, 6, dtype=torch.double, requires_grad=True)
    z0 = torch.randn(2, 7, dtype=torch.double)
    cand = z0.topk(3, 1).indices

    from torch.func import functional_call

    def f(t, e):  # differentiate w.r.t. the tokens AND the class embeddings
        return functional_call(rr, {"class_emb": e}, (z0, t), {"cand": cand})["logits"]

    e = rr.class_emb.detach().clone().requires_grad_(True)
    assert torch.autograd.gradcheck(f, (tok, e), eps=1e-6, atol=1e-5)


def test_all_class_evidence_shapes(batch):
    z0, tok, _ = batch
    ev = AllClassEvidence(C, D, d, H)
    out = ev(z0, tok)
    assert out["logits"].shape == (B, C)


def test_initial_reranker_is_close_to_global_head(batch):
    z0, tok, y = batch
    rr = CandidateReRanker(C, D, d, H, K, "rival").eval()
    out = rr(z0, tok)
    assert (out["logits"] - z0).abs().max() < 0.05   # small embeddings -> starts near z0


# ---- the pairwise variant -------------------------------------------------------------------

def test_rival_index_lists_every_other_position():
    idx = rival_index(4)
    assert idx.shape == (4, 3)
    for c in range(4):
        assert sorted(idx[c].tolist()) == [r for r in range(4) if r != c]


def test_pairwise_queries_are_antisymmetric():
    torch.manual_seed(6)
    emb = torch.randn(C, d)
    cand = torch.stack([torch.randperm(C)[:K] for _ in range(B)])
    q = pairwise_queries(emb, cand)                       # (B, K, K-1, d)
    assert q.shape == (B, K, K - 1, d)
    idx = rival_index(K)
    for c in range(K):
        for j in range(K - 1):
            r = int(idx[c, j])
            jr = int((idx[r] == c).nonzero())
            assert torch.allclose(q[:, c, j], -q[:, r, jr])
            assert torch.allclose(q[:, c, j], emb[cand[:, c]] - emb[cand[:, r]])


def test_soft_min_limits_and_bound():
    torch.manual_seed(7)
    s = torch.randn(B, K, K - 1)
    assert torch.allclose(soft_min(s, 0.0), s.min(-1).values)
    assert torch.allclose(soft_min(s, 1e-4), s.min(-1).values, atol=1e-3)
    assert torch.allclose(soft_min(s, 1e4), s.mean(-1), atol=1e-3)
    flat = torch.full((B, K, K - 1), 0.7)
    assert torch.allclose(soft_min(flat, 1.0), torch.full((B, K), 0.7))
    # Jensen: the soft-min never exceeds the mean, and never falls below the minimum
    for t in (0.1, 1.0, 10.0):
        v = soft_min(s, t)
        assert (v <= s.mean(-1) + 1e-6).all() and (v >= s.min(-1).values - 1e-6).all()


def test_pairwise_with_two_candidates_equals_rival():
    """K = 2: q_ab = e_a - e_b is both the pairwise and the rival query, and the soft-min
    over a single rival is the identity, so the two modes must agree exactly."""
    torch.manual_seed(8)
    z0, tok = torch.randn(B, C), torch.randn(B, N, D)
    rival = CandidateReRanker(C, D, d, H, 2, "rival", gt_insert=False).eval()
    pair = CandidateReRanker(C, D, d, H, 2, "pairwise", gt_insert=False).eval()
    pair.load_state_dict(rival.state_dict())
    with torch.no_grad():
        rival.class_emb.normal_()
        pair.class_emb.copy_(rival.class_emb)
    assert torch.allclose(rival(z0, tok)["logits"], pair(z0, tok)["logits"], atol=1e-5)


def test_pairwise_attention_shape_and_rival_dependence(batch):
    z0, tok, _ = batch
    torch.manual_seed(9)
    rr = CandidateReRanker(C, D, d, H, 3, "pairwise", gt_insert=False).eval()
    with torch.no_grad():
        rr.class_emb.normal_()
    out = rr(z0, tok, return_attn=True)
    assert out["attn"].shape == (B, H, 3, 2, N)
    assert out["pair_scores"].shape == (B, 3, 2)
    assert torch.allclose(out["attn"].sum(-1), torch.ones(B, H, 3, 2), atol=1e-5)
    c1 = torch.tensor([[0, 1, 2]] * B)
    c2 = torch.tensor([[0, 5, 6]] * B)
    s1 = rr(torch.zeros(B, C), tok, cand=c1)["scores"][:, 0]
    s2 = rr(torch.zeros(B, C), tok, cand=c2)["scores"][:, 0]
    assert not torch.allclose(s1, s2, atol=1e-6)


def test_pairwise_finite_difference_gradcheck():
    torch.manual_seed(10)
    rr = CandidateReRanker(7, 6, 8, 2, 3, "pairwise", gt_insert=False, pair_temp=0.5)
    rr = rr.double().eval()
    tok = torch.randn(2, 5, 6, dtype=torch.double, requires_grad=True)
    z0 = torch.randn(2, 7, dtype=torch.double)
    cand = z0.topk(3, 1).indices

    from torch.func import functional_call

    def f(t, e):
        return functional_call(rr, {"class_emb": e}, (z0, t), {"cand": cand})["logits"]

    e = rr.class_emb.detach().clone().requires_grad_(True)
    assert torch.autograd.gradcheck(f, (tok, e), eps=1e-6, atol=1e-5)


def test_contrastive_queries_refuses_pairwise():
    with pytest.raises(ValueError):
        contrastive_queries(torch.randn(C, d), torch.zeros(B, K, dtype=torch.long), "pairwise")
