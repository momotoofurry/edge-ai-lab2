"""Self-checks for storage accounting and quantization: python test_common.py"""
import torch
import torch.nn as nn

from common import (gptq_quantize, quant_storage_bits, record_base, rtn_quantize_weight,
                    set_compressed_bits, whitened_svd)
from evaluate import compression_rate


def tiny_model():
    """Mimics the Qwen3 module names: model.layers.N.{self_attn,mlp}.*_proj"""
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = nn.Linear(8, 4, bias=False)
    layer.mlp = nn.Module()
    layer.mlp.down_proj = nn.Linear(4, 8, bias=False)
    root = nn.Module()
    root.model = nn.Module()
    root.model.layers = nn.ModuleList([layer])
    root.lm_head = nn.Linear(8, 16, bias=False)  # not a target: must be ignored
    return root


def test_storage_bits():
    # [4 x 8] weight, 4-bit, groups of 4 columns -> 2 groups/row, 8 groups total
    # weights: 4*8*4 = 128 ; per group: fp16 scale (16) + 4-bit zero (4) = 20 -> 160
    assert quant_storage_bits(4, 8, 4, group_size=4, sym=False) == 128 + 160
    # symmetric: no zero point -> 128 + 8*16
    assert quant_storage_bits(4, 8, 4, group_size=4, sym=True) == 128 + 128
    # per-channel (one group per row): 4 groups
    assert quant_storage_bits(4, 8, 3, group_size=-1, sym=False) == 4 * 8 * 3 + 4 * (16 + 3)
    # INT4 g128 asym on a real Qwen shape -> 4 + 20/128 bits/weight
    b = quant_storage_bits(9728, 2560, 4, group_size=128) / (9728 * 2560)
    assert abs(b - (4 + 20 / 128)) < 1e-9


def test_compression_rate():
    m = tiny_model()
    assert compression_rate(m) == 0.0  # nothing recorded yet
    base = record_base(m)
    assert base == (32 + 32) * 16  # lm_head excluded
    assert compression_rate(m) == 0.0
    # quantize only q_proj: 4-bit, g4, asym = 288 bits ; down_proj stays fp16 = 512 bits
    set_compressed_bits(m.model.layers[0].self_attn.q_proj, quant_storage_bits(4, 8, 4, 4))
    expected = 1 - (288 + 32 * 16) / base
    assert abs(compression_rate(m) - expected) < 1e-12
    # a replaced module (e.g. low-rank) is looked up by the same name
    lowrank = nn.Sequential(nn.Linear(4, 2, bias=False), nn.Linear(2, 8, bias=False))
    m.model.layers[0].mlp.down_proj = lowrank
    expected = 1 - (288 + (8 + 16) * 16) / base
    assert abs(compression_rate(m) - expected) < 1e-12


def test_rtn_matches_part1():
    # Part 1 Q1/Q2 example: 2-bit asym on [-0.25, 0.75] -> scale 1/3
    w = torch.tensor([[-0.25, 0.0, 0.40, 0.75]])
    q = rtn_quantize_weight(w, 2, group_size=-1, sym=False)
    s = 1 / 3
    assert torch.allclose(q, torch.tensor([[-s, 0.0, s, 2 * s]]), atol=1e-3)


def test_gptq_identity_hessian_is_rtn():
    # with H = I there is no cross-column error compensation -> GPTQ == RTN
    torch.manual_seed(0)
    w = torch.randn(16, 64)
    for gs, sym in [(-1, False), (16, False), (16, True)]:
        g = gptq_quantize(w, torch.eye(64), 3, group_size=gs, sym=sym, actorder=False, percdamp=0.0)
        r = rtn_quantize_weight(w, 3, group_size=gs, sym=sym)
        assert torch.allclose(g, r, atol=1e-5), (gs, sym)


def test_gptq_beats_rtn_on_correlated_inputs():
    torch.manual_seed(0)
    x = torch.randn(2048, 64) @ torch.randn(64, 64)  # correlated features
    H = 2 * x.t() @ x / x.shape[0]
    w = torch.randn(32, 64)
    err = lambda q: ((x @ (w - q).t()) ** 2).mean().item()
    g = gptq_quantize(w, H, 3, group_size=16)
    r = rtn_quantize_weight(w, 3, group_size=16)
    assert err(g) < err(r)


def test_whitened_svd():
    torch.manual_seed(0)
    x = torch.randn(4096, 32) @ torch.randn(32, 32)
    H = 2 * x.t() @ x / x.shape[0]
    w = torch.randn(24, 32)
    B, A = whitened_svd(w, H, rank=24, percdamp=0.0)
    assert torch.allclose(A @ B, w, atol=1e-3)  # full rank reconstructs W
    B, A = whitened_svd(w, H, rank=8, percdamp=0.0)
    U, s, Vh = torch.linalg.svd(w, full_matrices=False)
    plain = U[:, :8] * s[:8] @ Vh[:8]
    err = lambda q: ((x @ (w - q).t()) ** 2).mean().item()
    assert err(A @ B) < err(plain)  # whitening minimises output error, not weight error


def test_prune_mlp_compensation():
    from case4 import prune_mlp
    torch.manual_seed(0)
    hidden, inter, keep = 16, 64, 40
    mlp = nn.Module()
    mlp.gate_proj = nn.Linear(hidden, inter, bias=False)
    mlp.up_proj = nn.Linear(hidden, inter, bias=False)
    mlp.down_proj = nn.Linear(inter, hidden, bias=False)
    a = torch.randn(4096, inter) @ (torch.eye(inter) + 0.5 * torch.randn(inter, inter) / 8)  # correlated acts
    H = 2 * a.t() @ a / a.shape[0]
    Wd = mlp.down_proj.weight.detach().clone()
    gate = mlp.gate_proj.weight.detach().clone()
    prune_mlp(mlp, H, keep, percdamp=0.0)
    assert mlp.gate_proj.weight.shape == (keep, hidden) and mlp.up_proj.weight.shape == (keep, hidden)
    assert mlp.down_proj.weight.shape == (hidden, keep)
    # kept gate rows are a subset of the original rows, in the original order
    idx = [int((gate == r).all(dim=1).nonzero()) for r in mlp.gate_proj.weight]
    assert idx == sorted(idx)
    ref = a @ Wd.t()
    comp = a[:, idx] @ mlp.down_proj.weight.t()
    naive = a[:, idx] @ Wd[:, idx].t()
    assert ((ref - comp) ** 2).mean() < ((ref - naive) ** 2).mean()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
