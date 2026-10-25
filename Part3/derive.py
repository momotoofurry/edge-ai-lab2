"""AdaptQuant threshold derivation (Algorithm 1), run once offline on a reference model;
the thresholds are then applied unchanged to Qwen3-4B."""
import argparse
import json
import sys

import torch
import torch.nn as nn

from mx import (EFF_BITS, STAT_MAX, mx_quantize, nvfp4_quantize, tile_errors, tile_stat)

PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
FMTS = ("fp4", "fp6", "fp8")
STEP = 0.02
GRID = [round(i * STEP, 2) for i in range(int(round(STAT_MAX / STEP)) + 1)]
M = len(GRID)  # threshold index M = CAP ("never promote")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class Histogram:
    """Per grid position: tile count and summed FP4/FP6/FP8/NVFP4 errors."""

    def __init__(self, device):
        self.h = torch.zeros(5, M, dtype=torch.float64, device=device)

    @torch.no_grad()
    def add(self, x, wc):
        pos = (tile_stat(x) / STEP + 1e-6).floor().clamp(0, M - 1).long().flatten()
        self.h[0] += torch.bincount(pos, minlength=M).double()
        for k, f in enumerate(FMTS, start=1):
            e = tile_errors(x, wc, mx_quantize(x, f)).flatten().double()
            self.h[k] += torch.bincount(pos, weights=e, minlength=M)
        e = tile_errors(x, wc, nvfp4_quantize(x)).flatten().double()
        self.h[4] += torch.bincount(pos, weights=e, minlength=M)


def solve(hists, slack_db=0.0):
    """Exact grid optimum of Eq. 10 for one operand and projection type (one histogram per layer)."""
    H = torch.stack(hists).double().cpu()
    zero = torch.zeros(H.shape[0], H.shape[1], 1, dtype=H.dtype)
    P = torch.cat([zero, H.cumsum(dim=2)], dim=2)
    N = P[:, 0].sum(dim=0)
    idx = torch.arange(M + 1)
    i, j = idx[:, None], idx[None, :]
    valid = i <= j
    total = N[M]
    bits = (EFF_BITS["fp4"] * N[i] + EFF_BITS["fp6"] * (N[j] - N[i]) + EFF_BITS["fp8"] * (total - N[j])) / total
    E4, E6, E8, NV = P[:, 1], P[:, 2], P[:, 3], P[:, 4]
    err = E4[:, i] + (E6[:, j] - E6[:, i]) + (E8[:, M][:, None, None] - E8[:, j])
    ref = NV[:, M][:, None, None]
    feasible = valid & (err <= ref * 10 ** (slack_db / 10)).all(dim=0)
    if not feasible.any():
        raise RuntimeError("no threshold pair meets the NVFP4 reference")
    # ties go to more MXFP4 (larger i)
    cost = torch.where(feasible, bits, torch.full_like(bits, float("inf")))
    best = cost.min()
    cand = (cost <= best + 1e-12).nonzero()
    bi, bj = max(cand.tolist(), key=lambda p: (p[0], p[1]))
    n_layer = P[:, 0, M]
    sqnr = -10 * torch.log10(err[:, bi, bj] / n_layer)
    sqnr_nv = -10 * torch.log10(NV[:, M] / n_layer)
    t = lambda k: GRID[k] if k < M else "CAP"
    return dict(t1=t(bi), t2=t(bj), bits=round(best.item(), 4),
                fp6_share=round(((N[bj] - N[bi]) / total).item(), 4),
                fp8_share=round(((total - N[bj]) / total).item(), 4),
                min_margin_db=round((sqnr - sqnr_nv).min().item(), 3), layers=H.shape[0])


class _Catcher(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.attention_type = getattr(layer, "attention_type", None)
        self.inp, self.kwargs = None, None

    def forward(self, hidden_states, **kwargs):
        self.inp, self.kwargs = hidden_states, kwargs
        raise ValueError


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.1-8B")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--out", default="thresholds.json")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="cpu",
                                                 low_cpu_mem_usage=True)
    model.eval()
    model.config.use_cache = False
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"])
    ids = tok(text, return_tensors="pt").input_ids[:, : args.seqlen].to(dev)

    inner = model.model
    inner.embed_tokens.to(dev)
    inner.rotary_emb.to(dev)
    catcher = _Catcher(inner.layers[0])
    inner.layers[0] = catcher
    try:
        model(ids)
    except ValueError:
        pass
    inner.layers[0] = catcher.layer
    x, kwargs = catcher.inp, {k: v for k, v in catcher.kwargs.items() if k not in ("past_key_values", "use_cache")}

    hist = {(op, p): [] for op in ("act", "wt") for p in PROJ}
    for li, layer in enumerate(inner.layers):
        layer.to(dev)
        lins = {p: getattr(layer.self_attn if p in PROJ[:4] else layer.mlp, p) for p in PROJ}
        moment, handles = {}, []

        def make_hook(p, lin):
            wc_act = lin.weight.float().pow(2).sum(dim=0)  # ||W[:, c]||^2

            def hook(module, inp):
                a = inp[0].reshape(-1, inp[0].shape[-1])
                moment[p] = a.float().pow(2).mean(dim=0)  # diag(X^T X) / T
                h = Histogram(dev)
                h.add(a, wc_act)
                hist[("act", p)].append(h.h)
            return hook

        for p, lin in lins.items():
            handles.append(lin.register_forward_pre_hook(make_hook(p, lin)))
        out = layer(x, **kwargs)
        x = out[0] if isinstance(out, tuple) else out
        for hd in handles:
            hd.remove()
        for p, lin in lins.items():
            h = Histogram(dev)
            h.add(lin.weight.data, moment[p])
            hist[("wt", p)].append(h.h)
        layer.to("cpu")
        log(f"[derive] layer {li} done")

    result = {"reference_model": args.model, "corpus": f"wikitext-2-raw-v1 train, first {args.seqlen} tokens",
              "grid_step": STEP, "slack_db": 0.0, "weight": {}, "act": {}}
    for (op, p), hs in hist.items():
        r = solve(hs)
        result["weight" if op == "wt" else "act"][p] = r
        log(f"[derive] {op:3s} {p:9s} {r}")
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    log(f"[derive] wrote {args.out}")


if __name__ == "__main__":
    main()
