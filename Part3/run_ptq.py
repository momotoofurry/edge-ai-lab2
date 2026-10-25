"""Calibration-free W+A quantization of Qwen3-4B, then WikiText-2 PPL.
Modes: adaptquant (thresholds.json), or the uniform baselines mxfp4 / mxfp6 / nvfp4."""
import argparse
import json
import sys

import torch
import torch.nn as nn

from evaluate import eval_ppl_wikitext2, load_model
from mx import EFF_BITS, NVFP4_BITS, TILE_ROWS, adaptquant, nvfp4_quantize

PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
INF = float("inf")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def thresholds_for(mode, table, operand, proj):
    if mode == "nvfp4":
        return None, None
    if mode == "mxfp4":
        return INF, INF
    if mode == "mxfp6":
        return 0.0, INF
    r = table[operand][proj]
    as_float = lambda t: INF if t == "CAP" else float(t)
    return as_float(r["t1"]), as_float(r["t2"])


def fake_quant(x2d, mode, t1, t2):
    if mode == "nvfp4":
        return nvfp4_quantize(x2d).to(x2d.dtype), None
    return adaptquant(x2d, t1, t2)


def act_hook(mode, t1, t2):
    def hook(module, inp):
        x = inp[0]
        a = x.reshape(-1, x.shape[-1])
        pad = (-a.shape[0]) % TILE_ROWS
        if pad:
            a = torch.cat([a, a.new_zeros(pad, a.shape[1])])
        q, _ = fake_quant(a, mode, t1, t2)
        return (q[: a.shape[0] - pad].reshape(x.shape),) + tuple(inp[1:])
    return hook


@torch.no_grad()
def quantize_model(model, mode, table):
    counts = {"fp4": 0, "fp6": 0, "fp8": 0}
    n_tiles = 0
    for layer in model.model.layers:
        for p in PROJ:
            lin = getattr(layer.self_attn if p in PROJ[:4] else layer.mlp, p)
            assert isinstance(lin, nn.Linear)
            tw = thresholds_for(mode, table, "weight", p)
            q, fmt = fake_quant(lin.weight.data, mode, *tw)
            lin.weight.data.copy_(q)
            if fmt is not None:
                for fid, name in enumerate(("fp4", "fp6", "fp8")):
                    counts[name] += int((fmt == fid).sum())
                n_tiles += fmt.numel()
            lin.register_forward_pre_hook(act_hook(mode, *thresholds_for(mode, table, "act", p)))
    if mode == "nvfp4":
        return dict(weight_bits=NVFP4_BITS)
    share = {k: v / n_tiles for k, v in counts.items()}
    bits = sum(share[k] * EFF_BITS[k] for k in share)
    # + per-tile format table, 5 B per 512 elements
    return dict(weight_bits=round(bits, 4), weight_bits_with_index=round(bits + 5 * 8 / 512, 4),
                tile_share={k: round(v, 4) for k, v in share.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="adaptquant", choices=["adaptquant", "mxfp4", "mxfp6", "nvfp4"])
    ap.add_argument("--thresholds", default="thresholds.json")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    args = ap.parse_args()

    table = None
    if args.mode == "adaptquant":
        with open(args.thresholds) as f:
            table = json.load(f)
    model, tok = load_model(args.model, "float16")
    info = quantize_model(model, args.mode, table)
    log(f"[ptq] mode={args.mode} {json.dumps(info)}")
    ppl = eval_ppl_wikitext2(model, tok)
    print(f"mode: {args.mode}")
    print(f"weight bits/element: {info['weight_bits']}")
    print(f"ppl: {ppl:.4f}", flush=True)


if __name__ == "__main__":
    main()
