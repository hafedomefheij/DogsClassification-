"""Evaluation repeats exactly: the random control's rival draws are seeded, the global RNG is untouched."""
import numpy as np
import torch

from dogbreed.models.heads import build_head
from dogbreed.training.cached import SplitData
from dogbreed.training.train_heads import predict


def _data(n=64, dg=12, t=9, dt=8, c=20):
    g = torch.Generator().manual_seed(0)
    return SplitData(ids=np.arange(n), y=torch.randint(0, c, (n,), generator=g),
                     g=[torch.randn(n, dg, generator=g)], tok=[torch.randn(n, t, dt, generator=g)],
                     device=torch.device("cpu"))


def test_random_control_is_reproducible_with_a_seed():
    torch.manual_seed(0)
    head = build_head({"name": "rerank", "mode": "random", "k": 5}, 12, 8, 20)
    with torch.no_grad():                       # make the evidence term matter for the final logits
        for p in head.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    data = _data()
    a = predict(head, data, torch.device("cpu"), seed=3)["logits"]
    state = torch.get_rng_state()
    b = predict(head, data, torch.device("cpu"), seed=3)["logits"]
    assert torch.equal(state, torch.get_rng_state())          # the global RNG was not advanced
    np.testing.assert_array_equal(a, b)
    c = predict(head, data, torch.device("cpu"), seed=4)["logits"]
    assert not np.array_equal(a, c)                           # other draws -> other rivals


def test_deterministic_heads_ignore_the_seed():
    torch.manual_seed(0)
    head = build_head({"name": "rerank", "mode": "rival", "k": 5}, 12, 8, 20)
    data = _data()
    a = predict(head, data, torch.device("cpu"), seed=3)["logits"]
    b = predict(head, data, torch.device("cpu"))["logits"]
    np.testing.assert_array_equal(a, b)
