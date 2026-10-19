"""stream.py: routing, grouping, one layer's phases, logits row mapping, stream save/load (CPU, no checkpoint)."""
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from step5q.refmodel import window_attention  # noqa: E402
from step5q.stream import Streamer, SeqState, group_sequences, route_topk  # noqa: E402


def test_route_topk_by_hand():
    """Bias selects expert 2 although its probability is lowest; its weight is the unbiased probability."""
    probs = [0.9, 0.8, 0.1, 0.2]
    bias = torch.tensor([0.0, 0.0, 1.0, 0.0])
    gate_w = torch.zeros(4, 3, dtype=torch.float64)
    gate_w[:, 0] = torch.tensor([math.log(p / (1 - p)) for p in probs], dtype=torch.float64)
    x = torch.tensor([[1.0, 0.0, 0.0], [1.0, 5.0, -2.0]])
    w, ids = route_topk(x, gate_w, bias, top_k=2, scaling=3.0)
    assert ids.dtype == torch.int64 and w.dtype == torch.float32
    assert ids.tolist() == [[2, 0], [2, 0]]
    want = [3.0 * 0.1 / (0.1 + 0.9 + 1e-20), 3.0 * 0.9 / (0.1 + 0.9 + 1e-20)]
    torch.testing.assert_close(w, torch.tensor([want, want]), rtol=1e-6, atol=1e-6)
    w2, _ = route_topk(x, gate_w, bias, top_k=2, scaling=3.0, renormalize=False)
    torch.testing.assert_close(w2[0], torch.tensor([0.3, 2.7]), rtol=1e-6, atol=1e-6)


def test_group_sequences():
    assert group_sequences([3, 4, 10, 2, 2], 7) == [[0, 1], [2], [3, 4]]
    assert group_sequences([5], 1) == [[0]]


def bare_streamer(resids, max_group_tokens=1 << 16):
    s = Streamer.__new__(Streamer)
    s.device = torch.device("cpu")
    s.cfg = SimpleNamespace(hidden_size=resids[0].shape[1], fp32_residual_connection=True)
    s.states = [SeqState(r.clone()) for r in resids]
    s.lengths = [r.shape[0] for r in resids]
    s.offsets = [sum(s.lengths[:i]) for i in range(len(resids))]
    s.n_tokens = sum(s.lengths)
    s.max_group_tokens = max_group_tokens
    s.layer = 0
    return s


class FakeAttn(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.scaling, self.sliding_window = 0.25, 2
        self.o_proj = nn.Linear(H, H, bias=False, dtype=torch.bfloat16)

    def _project(self, h, pos):
        T = h.shape[0]
        return h.view(T, 2, -1), h.view(T, 2, -1)[:, :1], h.view(T, 2, -1)[:, 1:]

    def _gate_and_project(self, o, h):
        return self.o_proj(o.reshape(o.shape[0], -1))


class FakeLayer(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.layer_idx, self.moe = 0, None
        self.self_attn = FakeAttn(H)
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()


def test_run_layer_phases_and_hooks():
    g = torch.Generator().manual_seed(0)
    H = 8
    resids = [torch.randn(L, H, generator=g) for L in (3, 4, 5)]
    s = bare_streamer(resids, max_group_tokens=7)
    layer = FakeLayer(H)
    calls, attn_calls = [], []

    def hook(l, x, ids, w, base):
        calls.append((l, x.shape[0], ids, w, base))

    def attn_hook(l, i, h, o_in):
        attn_calls.append((i, h.shape, o_in.shape))

    s.run_layer(layer, moe=lambda x, lay: x.float() * 0.5, hook=hook, attn_hook=attn_hook)
    assert s.layer == 1
    assert [(c[1], c[4]) for c in calls] == [(7, 0), (5, 7)] and calls[0][2] is None
    assert [c[0] for c in attn_calls] == [0, 1, 2]
    for r, st in zip(resids, s.states):
        h = r.to(torch.bfloat16)
        T = h.shape[0]
        o = window_attention(h.view(T, 2, -1), h.view(T, 2, -1)[:, :1], h.view(T, 2, -1)[:, 1:], 0.25, 2)
        r1 = r + layer.self_attn.o_proj(o.reshape(T, -1)).float()
        torch.testing.assert_close(st.resid, r1 + r1.to(torch.bfloat16).float() * 0.5, rtol=0, atol=0)
    with pytest.raises(ValueError):
        s.run_layer(layer)                                   # stream is at layer 1 now


def test_final_logits_row_mapping():
    g = torch.Generator().manual_seed(1)
    resids = [torch.randn(L, 8, generator=g) for L in (2, 3, 4)]
    s = bare_streamer(resids)
    head = nn.Linear(8, 5, bias=False, dtype=torch.bfloat16)
    rows = [8, 0, 3, 2, 4]
    got = s.final_logits(nn.Identity(), head, rows)
    flat = torch.cat(resids)
    want = head(flat[rows].to(torch.bfloat16)).float()
    assert torch.equal(got, want)
    with pytest.raises(IndexError):
        s.final_logits(nn.Identity(), head, [9])


def test_save_load_round_trip(tmp_path):
    pytest.importorskip("safetensors")
    g = torch.Generator().manual_seed(2)
    resids = [torch.randn(L, 8, generator=g) for L in (2, 3)]
    s = bare_streamer(resids)
    s.layer = 5
    s.save(str(tmp_path))
    s.layer = 6
    p6 = s.save(str(tmp_path))
    assert sorted(f.name for f in tmp_path.iterdir()) == ["stream_L06.safetensors"]
    t = bare_streamer([torch.zeros(2, 8), torch.zeros(3, 8)])
    assert t.load(str(tmp_path)) == 6 and t.layer == 6
    for r, st in zip(resids, t.states):
        assert st.resid.dtype == torch.float32
        assert torch.equal(st.resid, r.to(torch.bfloat16).float())
    with pytest.raises(ValueError):
        bare_streamer([torch.zeros(3, 8), torch.zeros(2, 8)]).load(str(tmp_path))
    assert p6.endswith("stream_L06.safetensors")
