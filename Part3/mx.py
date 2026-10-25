"""MX / NVFP4 fake quantizers and AdaptQuant's per-tile format selection.
An operand is [R, K] with K the reduction axis; a block is 1x32 along K sharing one E8M0
scale, a tile is 16x32 and gets one element format."""
import math

import torch

ELEM = {  # (mantissa bits, exponent bias, largest magnitude)
    "fp4": (1, 1, 6.0),     # E2M1
    "fp6": (3, 1, 7.5),     # E2M3
    "fp8": (3, 7, 448.0),   # E4M3
}
EFF_BITS = {"fp4": 4.25, "fp6": 6.25, "fp8": 8.25}  # payload + 8-bit scale / 32 elements
NVFP4_BITS = 4.5
BLOCK, TILE_ROWS = 32, 16
STAT_MAX = 0.5 * math.log2(BLOCK * TILE_ROWS)


def fp_round(x, fmt):
    """Round-to-nearest-even on the element grid, clamped to the largest magnitude."""
    mbits, bias, vmax = ELEM[fmt]
    a = x.abs()
    e = torch.floor(torch.log2(a.clamp(min=1e-30))).clamp(min=1 - bias)
    step = torch.exp2(e - mbits)
    q = (torch.round(a / step) * step).clamp(max=vmax)
    return torch.copysign(q, x)


def mx_quantize(x, fmt):
    shape = x.shape
    xb = x.float().reshape(*shape[:-1], shape[-1] // BLOCK, BLOCK)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    emax = math.floor(math.log2(ELEM[fmt][2]))
    sigma = torch.floor(torch.log2(amax.clamp(min=1e-30))) - emax
    s = torch.where(amax > 0, torch.exp2(sigma), torch.ones_like(amax))
    return (fp_round(xb / s, fmt) * s).reshape(shape)


def nvfp4_quantize(x):
    """E2M1 elements, 16-element blocks, E4M3 block scale under a per-tensor FP32 scale."""
    shape = x.shape
    xf = x.float()
    g = (xf.abs().amax() / (6.0 * 448.0)).clamp(min=1e-30)
    xb = xf.reshape(*shape[:-1], shape[-1] // 16, 16)
    sb = fp_round(xb.abs().amax(dim=-1, keepdim=True) / 6.0 / g, "fp8")
    s = torch.where(sb > 0, sb * g, torch.ones_like(sb))
    return (fp_round(xb / s, "fp4") * s).reshape(shape)


def to_tiles(x):
    """[R, K] -> [R/16, K/32, 16, 32]"""
    R, K = x.shape
    return x.reshape(R // TILE_ROWS, TILE_ROWS, K // BLOCK, BLOCK).transpose(1, 2)


def tile_stat(x):
    """log2(amax / rms) per tile."""
    t = to_tiles(x.float())
    amax = t.abs().amax(dim=(-1, -2))
    rms = t.pow(2).mean(dim=(-1, -2)).sqrt()
    return torch.where(amax > 0, torch.log2(amax / rms.clamp(min=1e-30)), torch.zeros_like(amax))


def tile_formats(stat, t1, t2=float("inf")):
    """0 = FP4, 1 = FP6, 2 = FP8"""
    return (stat >= t1).long() + (stat >= t2).long()


def adaptquant(x, t1, t2=float("inf")):
    """Returns (dequantized operand, per-tile format ids)."""
    fmt = tile_formats(tile_stat(x), t1, t2)
    out = mx_quantize(x, "fp4")
    for fid, name in ((1, "fp6"), (2, "fp8")):
        if (fmt == fid).any():
            sel = fmt == fid
            q = mx_quantize(x, name)
            mask = sel[:, None, :, None].expand(-1, TILE_ROWS, -1, BLOCK).reshape(x.shape)
            out = torch.where(mask, q, out)
    return out.to(x.dtype), fmt


def tile_errors(x, wc, quantized):
    """Per-tile noise-to-signal ratio, each element weighted by its column weight wc."""
    w = wc.float()[None, :]
    num = to_tiles((x.float() - quantized.float()).pow(2) * w).sum(dim=(-1, -2))
    den = to_tiles(x.float().pow(2) * w).sum(dim=(-1, -2))
    return num / den.clamp(min=1e-30)
