"""Case 4: structured pruning of MLP intermediate channels + GPTQ. Pruned matrices really
shrink, so no mask/index is stored."""
import torch
import torch.nn as nn

from common import (capture_layer0_inputs, collect_hessians, get_calib_data, gptq_module, log, record_base,
                    run_layer, seed_everything)

PRUNE_RATIO = 0.30  # per layer, of the intermediate channels
ATTN_QUANT = dict(wbits=3, group_size=512, sym=False, mse=True)
MLP_QUANT = dict(wbits=3, group_size=512, sym=False, mse=True)
MIN_PRUNE_RATE = 0.20
NSAMPLES = 128
SEED = 0


def _linear(weight):
    out_f, in_f = weight.shape
    m = nn.Linear(in_f, out_f, bias=False, device=weight.device, dtype=weight.dtype)
    m.weight.data.copy_(weight)
    return m


@torch.no_grad()
def prune_mlp(mlp, H_down, keep_n, percdamp=0.01):
    """Keep the channels with the largest ||W_down[:, j]||^2 * H_jj and refit down_proj:
    W' = W_keep + W_rm H_rk H_kk^-1 (least squares, no training)."""
    Wd = mlp.down_proj.weight.double()
    H = H_down.double()
    score = (Wd ** 2).sum(dim=0) * torch.diag(H)
    keep = torch.topk(score, keep_n).indices.sort().values
    mask = torch.ones(Wd.shape[1], dtype=torch.bool, device=Wd.device)
    mask[keep] = False
    rm = mask.nonzero().flatten()

    Hkk = H[keep][:, keep]
    Hkk += percdamp * torch.mean(torch.diag(Hkk)) * torch.eye(keep_n, device=H.device, dtype=H.dtype)
    rhs = Wd[:, rm] @ H[rm][:, keep]
    Wd_new = Wd[:, keep] + torch.linalg.solve(Hkk, rhs.t()).t()

    dtype = mlp.down_proj.weight.dtype
    mlp.gate_proj = _linear(mlp.gate_proj.weight[keep])
    mlp.up_proj = _linear(mlp.up_proj.weight[keep])
    mlp.down_proj = _linear(Wd_new.to(dtype))


@torch.no_grad()
def compress(model, tokenizer, rate=0.85, wbits=None):
    seed_everything(SEED)
    record_base(model)
    base_numel = sum(model._base_linear_numel.values())
    inter = model.config.intermediate_size
    keep_n = inter - int(round(PRUNE_RATIO * inter))

    calib = get_calib_data(tokenizer, nsamples=NSAMPLES, seed=SEED)
    inps, kwargs = capture_layer0_inputs(model, calib)
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        Hs = collect_hessians(layer, ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], inps, kwargs)
        for n in ("q_proj", "k_proj", "v_proj"):
            gptq_module(getattr(attn, n), Hs[f"self_attn.{n}"], ATTN_QUANT)
        H = collect_hessians(layer, ["self_attn.o_proj"], inps, kwargs)["self_attn.o_proj"]
        gptq_module(attn.o_proj, H, ATTN_QUANT)

        Hs = collect_hessians(layer, ["mlp.gate_proj", "mlp.down_proj"], inps, kwargs)
        prune_mlp(layer.mlp, Hs["mlp.down_proj"], keep_n)
        gptq_module(layer.mlp.gate_proj, Hs["mlp.gate_proj"], MLP_QUANT)
        gptq_module(layer.mlp.up_proj, Hs["mlp.gate_proj"], MLP_QUANT)  # shares gate_proj's input
        del Hs
        H = collect_hessians(layer, ["mlp.down_proj"], inps, kwargs)["mlp.down_proj"]
        gptq_module(layer.mlp.down_proj, H, MLP_QUANT)
        del H

        inps = run_layer(layer, inps, kwargs)
        log(f"[case4] layer {li} done")

    numel = sum(model.get_submodule(n).weight.numel() for n in model._base_linear_numel)
    model._prune_rate = 1 - numel / base_numel
    log(f"[case4] pruning rate = {model._prune_rate:.4%}")
    assert model._prune_rate >= MIN_PRUNE_RATE, f"pruning rate {model._prune_rate:.2%} < {MIN_PRUNE_RATE:.0%}"
    torch.cuda.empty_cache()
