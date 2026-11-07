"""Calibration-free W+A quantization of Qwen3-4B, then WikiText-2 PPL.
Modes: adaptquant (thresholds.json), or the uniform baselines mxfp4 / mxfp6 / nvfp4."""
import argparse
import json
import sys

import torch
import torch.nn as nn

from evaluate import eval_ppl_wikitext2, load_model
from mx import NVFP4_BITS, TILE_ROWS, adaptquant, bits_from_counts, nvfp4_quantize

PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
INF = float("inf")
BIT_LIMIT = 5.3  # (weight bits + activation bits) / 2 must not exceed this


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


def act_hook(mode, t1, t2, counts):
    """Quantizes the Linear's input on every forward and adds the chosen tile formats
    to `counts` (FP4 / FP6 / FP8 tile counts, shared by all hooks)."""
    def hook(module, inp):
        x = inp[0]
        a = x.reshape(-1, x.shape[-1])
        pad = (-a.shape[0]) % TILE_ROWS
        if pad:
            a = torch.cat([a, a.new_zeros(pad, a.shape[1])])
        q, fmt = fake_quant(a, mode, t1, t2)
        if fmt is not None:
            counts["act"] += torch.bincount(fmt.flatten(), minlength=3).double().to(counts["act"].device)
        return (q[: a.shape[0] - pad].reshape(x.shape),) + tuple(inp[1:])
    return hook


def operand_bits(counts, mode):
    if mode == "nvfp4":
        return NVFP4_BITS, None
    # only the mixed-precision mode needs a per-tile format tag
    return bits_from_counts(counts, tagged=(mode == "adaptquant"))


@torch.no_grad()
def quantize_model(model, mode, table):
    """Quantizes every target weight now and installs the activation hooks. Returns the tile
    counters: "wt" is final after this call, "act" fills up during evaluation."""
    dev = next(model.parameters()).device
    counts = {"wt": torch.zeros(3, dtype=torch.float64), "act": torch.zeros(3, dtype=torch.float64, device=dev)}
    for layer in model.model.layers:
        for p in PROJ:
            lin = getattr(layer.self_attn if p in PROJ[:4] else layer.mlp, p)
            assert isinstance(lin, nn.Linear)
            tw = thresholds_for(mode, table, "weight", p)
            q, fmt = fake_quant(lin.weight.data, mode, *tw)
            lin.weight.data.copy_(q)
            if fmt is not None:
                counts["wt"] += torch.bincount(fmt.flatten(), minlength=3).double().cpu()
            a1, a2 = thresholds_for(mode, table, "act", p)
            lin.register_forward_pre_hook(act_hook(mode, a1, a2, counts))
    return counts


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
    counts = quantize_model(model, args.mode, table)
    w_bits, w_share = operand_bits(counts["wt"], args.mode)
    log(f"[ptq] mode={args.mode} weight bits={w_bits:.4f} weight tile share={w_share}")
    ppl = eval_ppl_wikitext2(model, tok)
    a_bits, a_share = operand_bits(counts["act"], args.mode)
    log(f"[ptq] activation bits={a_bits:.4f} activation tile share={a_share}")
    rate = (w_bits + a_bits) / 2
    print(f"mode: {args.mode}")
    print(f"weight bits/element: {w_bits:.4f}")
    print(f"activation bits/element: {a_bits:.4f}")
    print(f"effective bit rate: {rate:.4f} (limit {BIT_LIMIT}: {'OK' if rate <= BIT_LIMIT else 'EXCEEDED'})")
    print(f"ppl: {ppl:.4f}", flush=True)


if __name__ == "__main__":
    main()
