"""refmodel.py: FP8 block dequant, windowed GQA attention, dense_attention wiring (CPU, no checkpoint)."""
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from step5q.refmodel import dense_attention, fp8_block_dequant, window_attention  # noqa: E402


def test_fp8_block_dequant_matches_loop():
    g = torch.Generator().manual_seed(0)
    O, I, B = 5, 7, 2                                    # ragged blocks: scale is 3 x 4
    w = (torch.randn(O, I, generator=g) * 4).to(torch.float8_e4m3fn)
    s = torch.rand(3, 4, generator=g) + 0.5
    ref = torch.empty(O, I)
    for o in range(O):
        for i in range(I):
            ref[o, i] = w[o, i].float() * s[o // B, i // B]
    got = fp8_block_dequant(w, s, block=B)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, ref.to(torch.bfloat16))


def test_fp8_block_dequant_stacked_128():
    g = torch.Generator().manual_seed(1)
    w = (torch.randn(3, 256, 384, generator=g) * 8).to(torch.float8_e4m3fn)
    s = torch.rand(3, 2, 3, generator=g) + 0.1
    got = fp8_block_dequant(w, s)
    for e, o, i in [(0, 0, 0), (1, 127, 128), (1, 128, 127), (2, 255, 383), (2, 130, 300)]:
        want = (w[e, o, i].float() * s[e, o // 128, i // 128]).to(torch.bfloat16)
        assert got[e, o, i] == want
    with pytest.raises(ValueError):
        fp8_block_dequant(w, s[:, :1])


def naive_attention(q, k, v, scale, window):
    """Per-head masked softmax: token i sees j with j <= i and i - j < window."""
    T, Hq, _ = q.shape
    rep = Hq // k.shape[1]
    i = torch.arange(T).view(-1, 1)
    j = torch.arange(T).view(1, -1)
    allowed = (j <= i) & ((i - j < window) if window else torch.ones(T, T, dtype=torch.bool))
    out = torch.empty_like(q)
    for h in range(Hq):
        s = (q[:, h] @ k[:, h // rep].T) * scale
        out[:, h] = torch.softmax(s.masked_fill(~allowed, -math.inf), -1) @ v[:, h // rep]
    return out


@pytest.mark.parametrize("Hq,Hkv", [(2, 1), (4, 2)])
@pytest.mark.parametrize("window", [None, 3])
def test_window_attention_matches_naive(Hq, Hkv, window):
    g = torch.Generator().manual_seed(2)
    T, D = 11, 8
    q, k, v = (torch.randn(T, h, D, generator=g) for h in (Hq, Hkv, Hkv))
    want = naive_attention(q, k, v, D ** -0.5, window)
    got = window_attention(q, k, v, D ** -0.5, window, chunk=4)   # chunk 4 crosses chunk boundaries
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_window_convention():
    """Window 3: token 5 attends to exactly 3, 4, 5; token 1 to 0, 1 (uniform weights with q = 0)."""
    T = 8
    q = torch.zeros(T, 2, T)
    k = torch.randn(T, 1, T)
    v = torch.eye(T).view(T, 1, T)
    out = window_attention(q, k, v, 1.0, 3, chunk=4)
    torch.testing.assert_close(out[5, 0], torch.tensor([0, 0, 0, 1, 1, 1, 0, 0]) / 3)
    torch.testing.assert_close(out[1, 1], torch.tensor([1, 1, 0, 0, 0, 0, 0, 0]) / 2)
    torch.testing.assert_close(out[7, 0], torch.tensor([0, 0, 0, 0, 0, 1, 1, 1]) / 3)


def test_dense_attention_wiring():
    """dense_attention = _project -> window_attention(scaling, sliding_window) -> _gate_and_project."""
    g = torch.Generator().manual_seed(3)
    T, Hq, D = 9, 2, 8
    hidden = torch.randn(T, Hq * D, generator=g)
    kv = torch.randn(T, 1, D, generator=g)
    pos = torch.arange(T, dtype=torch.int32)

    def project(h, p):
        assert torch.equal(p, pos)
        return h.view(T, Hq, D), kv, kv * 2

    attn = SimpleNamespace(scaling=0.3, sliding_window=3, _project=project,
                           _gate_and_project=lambda o, h: o.reshape(T, -1) * 2 + h)
    got = dense_attention(SimpleNamespace(self_attn=attn), hidden, pos)
    want = naive_attention(hidden.view(T, Hq, D), kv, kv * 2, 0.3, 3).reshape(T, -1) * 2 + hidden
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
