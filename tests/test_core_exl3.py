"""exl3.py: argument plumbing and the serving-path expert forward on CPU; encode/reconstruct on GPU only."""
import importlib.util

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from step5q import exl3  # noqa: E402

HAS_GPU = torch.cuda.is_available() and importlib.util.find_spec("exllamav3") is not None


def test_quant_args_and_h_data(tmp_path):
    qa = exl3.quant_args(5.0, 7, str(tmp_path))
    assert qa["K"] == 5 and type(qa["K"]) is int and qa["mul1"] is True and qa["seed"] == 7
    assert "no_refit" not in qa and exl3.quant_args(3, 0, str(tmp_path), no_refit=True)["no_refit"] is True
    with pytest.raises(ValueError):
        exl3.quant_args(2.5, 0, str(tmp_path))
    H = torch.eye(4)
    hd = exl3.h_data(H, 10, "k")
    assert hd["H"] is H and hd["count"] == 10 and hd["finalized"] is False and hd["num_total"] == 40


def test_encode_expert_rejects_k():
    with pytest.raises(ValueError):
        exl3.encode_expert(None, None, None, {}, {}, 6, 0, "/nonexistent")


def test_expert_forward_dense_semantics():
    g = torch.Generator().manual_seed(0)
    n, H, I = 6, 32, 16
    x = torch.randn(n, H, generator=g)
    w1, w3 = torch.randn(I, H, generator=g), torch.randn(I, H, generator=g)
    w2 = torch.randn(H, I, generator=g) * 0.1
    w1[0] *= 20                                                   # pushes silu(gate) past the 7.0 clamp
    gw = torch.rand(n, generator=g)
    y = exl3.expert_forward_dense(x, gw, w1, w3, w2)
    assert y.dtype == torch.float32
    xh = x.half()
    gate, up = F.linear(xh, w1.half()).float(), F.linear(xh, w3.half()).float()
    assert (F.silu(gate) > 7).any()
    a = (F.silu(gate).clamp(max=7.0) * up.clamp(-7.0, 7.0)).half()
    want = F.linear(a, w2.half()).float() * gw.half().float()[:, None]     # weight after down_proj
    torch.testing.assert_close(y, want, rtol=0, atol=0)
    y2 = exl3.expert_forward_dense(x, torch.full((n,), 2.0), w1, w3, w2)
    y1 = exl3.expert_forward_dense(x, torch.ones(n), w1, w3, w2)
    assert torch.equal(y2, 2 * y1)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GPU, reason="needs CUDA and exllamav3")
def test_encode_expert_reconstruct_gpu(tmp_path):
    g = torch.Generator(device="cuda").manual_seed(0)
    H, I, N = 512, 256, 4096
    w1, w3 = (torch.randn(I, H, device="cuda", generator=g) * 0.02 for _ in range(2))
    w2 = torch.randn(H, I, device="cuda", generator=g) * 0.02
    x = torch.randn(N, H, device="cuda", generator=g)
    a = F.silu(x @ w1.T).clamp(max=7) * (x @ w3.T).clamp(-7, 7)
    t, pe = exl3.encode_expert(w1, w3, w2, exl3.h_data(x.T @ x, N, "gu"), exl3.h_data(a.T @ a, N, "d"), 4, 0,
                               str(tmp_path))
    assert set(t) == {"w1", "w3", "w2"} and set(t["w1"]) >= {"trellis", "suh", "svh", "mul1"}
    assert t["w1"]["trellis"].shape == (H // 16, I // 16, 64)
    for proj, w, (i, o) in (("w1", w1, (H, I)), ("w3", w3, (H, I)), ("w2", w2, (I, H))):
        r = exl3.reconstruct(t[proj], i, o)
        assert r.shape == w.shape and r.dtype == torch.float16
        assert ((r.float() - w).norm() / w.norm()).item() < 0.15, proj
        assert 0 <= pe[proj] < 0.1
