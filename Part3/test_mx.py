"""Self-checks for the MX / NVFP4 quantizers: python test_mx.py"""
import math

import torch

from mx import STAT_MAX, adaptquant, fp_round, mx_quantize, nvfp4_quantize, tile_errors, tile_stat


def sqnr(x, q):
    return 10 * math.log10((x.pow(2).sum() / (x - q).pow(2).sum()).item())


def test_element_grids():
    x = torch.linspace(0, 8, 100001)
    fp4 = sorted(set(fp_round(x, "fp4").tolist()))
    assert fp4 == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    fp6 = sorted(set(fp_round(x, "fp6").tolist()))
    assert len(fp6) == 32 and fp6[1] == 0.125 and fp6[-1] == 7.5  # E2M3: 31 magnitudes + 0
    assert fp_round(torch.tensor([1e6]), "fp8").item() == 448.0
    assert fp_round(torch.tensor([2.0 ** -9]), "fp8").item() == 2.0 ** -9
    # flush below u_f / 2, round half to even on the grid
    assert fp_round(torch.tensor([0.24, 0.26]), "fp4").tolist() == [0.0, 0.5]
    assert fp_round(torch.tensor([2.5, 3.5, 5.0]), "fp4").tolist() == [2.0, 4.0, 4.0]


def test_mx_sqnr_relative_to_paper_table1():
    # Table 1 (k = 32, E8M0): FP4 17.33, FP6 29.31, FP8 29.39 dB. On Gaussian data every format
    # sits ~1.5 dB higher (lighter tails than model tensors), but the gaps must match:
    # FP6 - FP4 ~= 12 dB (two mantissa bits x 6.02 dB) and FP8 ~= FP6 (same mantissa width).
    torch.manual_seed(0)
    x = torch.randn(4096, 4096)
    s4, s6, s8 = (sqnr(x, mx_quantize(x, f)) for f in ("fp4", "fp6", "fp8"))
    assert 17.0 < s4 < 19.5, s4
    assert abs((s6 - s4) - (29.31 - 17.33)) < 0.5, (s4, s6)
    assert abs(s8 - s6) < 0.5, (s6, s8)
    assert sqnr(x, nvfp4_quantize(x)) > s4  # finer block + E4M3 scale beats MXFP4


def test_fp4_widens_exactly_to_fp6():
    # Appendix C.2: E2M1 codes are E2M3 codes and both use the same E8M0 scale
    torch.manual_seed(1)
    x = torch.randn(64, 256) * 3
    q4 = mx_quantize(x, "fp4")
    assert torch.equal(mx_quantize(q4, "fp6"), q4)


def test_tile_stat_and_selection():
    torch.manual_seed(2)
    x = torch.randn(32, 64)
    x[16:, 32:] = 0
    x[16, 40] = 50.0  # one spike in an otherwise empty tile -> maximal stat
    st = tile_stat(x)
    assert st.shape == (2, 2)
    assert abs(st[1, 1].item() - STAT_MAX) < 1e-5
    assert (st <= STAT_MAX + 1e-5).all() and (st >= 0).all()
    q, fmt = adaptquant(x, t1=3.0)
    assert fmt.tolist() == [[0, 0], [0, 1]]
    # the FP6 tile is quantized with FP6, the others with FP4
    assert torch.equal(q[16:, 32:], mx_quantize(x, "fp6")[16:, 32:])
    assert torch.equal(q[:16], mx_quantize(x, "fp4")[:16])


def test_tile_errors_weighting():
    torch.manual_seed(3)
    x = torch.randn(16, 64)
    q = mx_quantize(x, "fp4")
    e1 = tile_errors(x, torch.ones(64), q)
    e2 = tile_errors(x, torch.full((64,), 7.0), q)
    assert torch.allclose(e1, e2)  # a constant weight cancels in the ratio
    assert abs(-10 * math.log10(e1.mean().item()) - 17.3) < 1.5


def test_solve_threshold_program():
    from derive import M, Histogram, solve
    torch.manual_seed(4)
    x = torch.randn(64, 256)
    x[:16] *= torch.rand(16, 256) ** 8 * 20  # heavy-tailed rows -> high-stat tiles
    wc = torch.ones(256)
    h = Histogram("cpu")
    h.add(x, wc)
    loose = h.h.clone()
    loose[4] *= 100  # NVFP4 reference 20 dB worse -> all-FP4 is feasible
    r = solve([loose])
    assert r["t1"] == "CAP" and r["bits"] == 4.25
    tight = h.h.clone()
    tight[4] = tight[2] * 1.0001  # reference == FP6 everywhere -> everything must be FP6
    r = solve([tight])
    # every tile ends up FP6; among equal-cost thresholds the tie-break picks the largest t1
    # (here: the lowest occupied grid position), as the paper prescribes
    assert r["fp6_share"] == 1.0 and r["t2"] == "CAP" and abs(r["bits"] - 6.25) < 1e-9
    assert r["t1"] == min(i for i in range(M) if h.h[0, i] > 0) * 0.02
    r = solve([h.h])  # real NVFP4 reference: a mix, never FP8 here
    assert 4.25 < r["bits"] < 6.25 and r["min_margin_db"] >= 0 and r["t2"] == "CAP", r
    assert M == 226


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
