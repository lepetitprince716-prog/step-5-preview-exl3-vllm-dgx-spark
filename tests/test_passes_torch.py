"""Torch-side pass helpers on CPU with tiny tensors (skipped where torch is missing). step5q.exl3 is replaced by a fake
module where needed, so these tests do not depend on lane A's code or exllamav3."""
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p_torch  # noqa: E402


def fake_exl3(monkeypatch):
    """step5q.exl3 stand-in: expert_forward_dense with the README semantics (fp16 weights / activations, routing weight
    after down_proj, fp32 out); reconstruct reads a dense 'w' piece."""
    def expert_forward_dense(x, g, w1, w3, w2, limit=7.0):
        # elementwise only: row results cannot depend on the batch (matmul blocking would), so slot order is testable
        xh = x.to(torch.float16).float()
        h = F.silu(xh * w1.float()[0, 0]).clamp(max=limit) * (xh * w3.float()[0, 0]).clamp(-limit, limit)
        return torch.tanh(h * w2.float()[:, 0]) * g.float()[:, None]
    mod = types.ModuleType("step5q.exl3")
    mod.expert_forward_dense = expert_forward_dense
    mod.reconstruct = lambda t, i, o: t["trellis"]
    pkg = types.ModuleType("step5q")
    pkg.exl3 = mod
    monkeypatch.setitem(sys.modules, "step5q", pkg)
    monkeypatch.setitem(sys.modules, "step5q.exl3", mod)
    return mod


def routing(N, E, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(N)])
    w = torch.rand(N, k, generator=g) + 0.1
    return w / w.sum(-1, keepdim=True) * 3.0, ids


def test_split_out_roundtrip():
    g = torch.Generator().manual_seed(1)
    sizes = [256, 128, 128]
    t = {"trellis": torch.randint(-2 ** 15, 2 ** 15, (8, sum(sizes) // 16, 48), generator=g, dtype=torch.int32)
         .to(torch.int16), "suh": torch.randn(128, generator=g).half(), "svh": torch.randn(sum(sizes), generator=g)
         .half(), "mul1": torch.tensor(7, dtype=torch.int32)}
    parts = p_torch.split_out(t, sizes)
    assert torch.equal(torch.cat([p["trellis"] for p in parts], 1), t["trellis"])   # the plugin's fused concat
    assert torch.equal(torch.cat([p["svh"] for p in parts]), t["svh"])
    assert all(torch.equal(p["suh"], t["suh"]) and torch.equal(p["mul1"], t["mul1"]) for p in parts)
    with pytest.raises(ValueError):
        p_torch.split_out(t, [192, 320])


def test_native_slots_matches_official_cpu_fallback():
    N, E, H, I, k = 12, 6, 32, 16, 4
    g = torch.Generator().manual_seed(2)
    x = torch.randn(N, H, generator=g).bfloat16()
    W1, W3 = torch.randn(E, I, H, generator=g).bfloat16(), torch.randn(E, I, H, generator=g).bfloat16()
    W2 = torch.randn(E, H, I, generator=g).bfloat16()
    w, ids = routing(N, E, k)
    slot, out = p_torch.native_slots(x, w, ids, W1, W3, W2)
    ref_slot = torch.zeros(N, k, H, dtype=torch.bfloat16)          # inference/model.py Step5MoE.forward, CPU path
    for e in range(E):
        rows, slots = (ids == e).nonzero(as_tuple=True)
        if rows.numel():
            t = x[rows]
            act = (F.silu(F.linear(t, W1[e]).float()).clamp(max=7.0) * F.linear(t, W3[e]).float().clamp(-7, 7)).bfloat16()
            ref_slot[rows, slots] = F.linear(act, W2[e]).bfloat16()
    ref = torch.zeros(N, H)
    for s in range(k):
        ref = ref + ref_slot[:, s].float() * w[:, s].unsqueeze(-1)
    assert torch.equal(slot, ref_slot) and torch.equal(out, ref.bfloat16())


def test_quant_routed_sums_in_slot_order(monkeypatch):
    mod = fake_exl3(monkeypatch)
    N, E, H, I, k = 10, 5, 32, 16, 4
    g = torch.Generator().manual_seed(3)
    x = torch.randn(N, H, generator=g).bfloat16()
    cache = types.SimpleNamespace(w={e: (torch.randn(I, H, generator=g).half(), torch.randn(I, H, generator=g).half(),
                                         torch.randn(H, I, generator=g).half()) for e in range(E)})
    w, ids = routing(N, E, k, seed=4)
    out = p_torch.quant_routed(x, w, ids, cache)
    ref = torch.zeros(N, H)
    for t in range(N):
        acc = torch.zeros(H)
        for s in range(k):                                         # fp32, slot order
            e = int(ids[t, s])
            acc = acc + mod.expert_forward_dense(x[t:t + 1], w[t:t + 1, s], *cache.w[e])[0]
        ref[t] = acc
    assert torch.equal(out, ref)


def test_expert_cache_reads_sero_keys(monkeypatch):
    fake_exl3(monkeypatch)
    tens = {}
    for e in range(3):
        for p in ("gate_proj", "up_proj", "down_proj"):
            tens[f"model.layers.5.moe.experts.{e}.{p}.trellis"] = torch.zeros(2, 1, 32, dtype=torch.int16)
    with pytest.raises(SystemExit):
        p_torch.ExpertCache(5, tens, 4)                            # expert 3 missing


def test_p3_advance_adds_in_stream_order(monkeypatch):
    import p3_pass1
    monkeypatch.setattr(p3_pass1, "ROUTE_CHUNK", 3)
    lens = [4, 1, 5]
    states = [types.SimpleNamespace(resid=torch.zeros(n, 2)) for n in lens]
    s = types.SimpleNamespace(states=states)
    flat = torch.arange(sum(lens) * 2, dtype=torch.float32).view(-1, 2)
    p3_pass1.advance(s, flat, lambda x: x * 2)
    assert torch.equal(torch.cat([st.resid for st in states]), flat * 2)


def test_p3_assign_parts_balances_changed():
    import p3_pass1
    ks = [5, 2, 4, 3, 5, 2]
    part_of = p3_pass1.assign_parts(ks, [0, 1, 2, 4], 2)
    assert part_of[0] != part_of[4]                                # the two K5 re-encodes go to different boxes
    assert sorted(part_of) == list(range(6))


def test_p0_stats_counts(monkeypatch):
    import p0_stats
    N, E, H, I, k = 9, 6, 16, 16, 3
    g = torch.Generator().manual_seed(5)
    w, ids = routing(N, E, k, seed=6)
    moe = types.SimpleNamespace(gate_proj=types.SimpleNamespace(weight=torch.randn(E, I, H, generator=g).bfloat16()),
                                up_proj=types.SimpleNamespace(weight=torch.randn(E, I, H, generator=g).bfloat16()),
                                down_proj=types.SimpleNamespace(weight=torch.randn(E, H, I, generator=g).bfloat16()))
    layer = types.SimpleNamespace(moe=moe, share_expert=lambda x: torch.zeros_like(x))
    s = types.SimpleNamespace(route=lambda lay, x: (w, ids))
    st = p0_stats.LayerStats(E, "cpu", ["a", "b"])
    dom = torch.tensor([0] * 5 + [1] * 4)
    y = p0_stats.stats_moe(s, st, dom, [0])(torch.randn(N, H, generator=g).bfloat16(), layer)
    assert y.dtype == torch.float32 and y.shape == (N, H)
    assert st.v["count"].sum().item() == N * k and st.v["top1"].sum().item() == N
    rec = st.to_json()
    assert rec["domains"]["a"]["tokens"] == 5 and sum(rec["domains"]["b"]["count"]) == 4 * k
    assert abs(sum(rec["out_energy"]) - sum(rec["domains"]["a"]["out_energy"]) - sum(rec["domains"]["b"]["out_energy"])) \
        < 1e-6 * max(1.0, sum(rec["out_energy"]))
