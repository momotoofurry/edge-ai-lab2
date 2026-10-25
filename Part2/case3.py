"""Case 3: whitened SVD on the MLP + INT8. The shared rank is solved so the overall
ratio lands just above `rate`."""
import torch

from common import (FP16_BITS, SEQUENTIAL, LowRankLinear, capture_layer0_inputs, collect_hessians, full_name,
                    get_calib_data, iter_targets, log, lowrank_numel, quant_storage_bits, record_base,
                    rtn_quantize_weight, run_layer, seed_everything, set_compressed_bits, submodule_name,
                    whitened_svd)

SVD_TARGETS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
FACTOR_QUANT = dict(wbits=8, group_size=-1, sym=False)
OTHER_QUANT = {  # anything not listed stays FP16
    "self_attn.k_proj": dict(wbits=8, group_size=-1, sym=False),
    "self_attn.v_proj": dict(wbits=8, group_size=-1, sym=False),
}
MIN_SVD_RATE = 0.20
NSAMPLES = 128
SEED = 0


def _lowrank_bits(out_f, in_f, rank):
    q = FACTOR_QUANT
    return (quant_storage_bits(rank, in_f, q["wbits"], q["group_size"], q["sym"])
            + quant_storage_bits(out_f, rank, q["wbits"], q["group_size"], q["sym"]))


def _plan(model, rank):
    bits = numel = 0
    for full, m in iter_targets(model):
        name = submodule_name(full)
        if name in SVD_TARGETS:
            bits += _lowrank_bits(m.out_features, m.in_features, rank)
            numel += lowrank_numel(m.out_features, m.in_features, rank)
        elif name in OTHER_QUANT:
            q = OTHER_QUANT[name]
            bits += quant_storage_bits(m.out_features, m.in_features, q["wbits"], q["group_size"], q["sym"])
            numel += m.weight.numel()
        else:
            bits += m.weight.numel() * FP16_BITS
            numel += m.weight.numel()
    return bits, numel


def solve_rank(model, target_rate):
    base_bits = model._base_linear_bits
    lo, hi = 1, 2560
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if 1 - _plan(model, mid)[0] / base_bits >= target_rate:
            lo = mid
        else:
            hi = mid - 1
    return lo


@torch.no_grad()
def compress(model, tokenizer, rate=0.5, wbits=None):
    seed_everything(SEED)
    record_base(model)
    base_numel = sum(model._base_linear_numel.values())

    rank = solve_rank(model, rate)
    bits, numel = _plan(model, rank)
    svd_rate = 1 - numel / base_numel
    log(f"[case3] rank={rank} planned overall={1 - bits / model._base_linear_bits:.4%} svd_rate={svd_rate:.4%}")
    assert svd_rate >= MIN_SVD_RATE, f"SVD part removes only {svd_rate:.2%} < {MIN_SVD_RATE:.0%}"
    model._svd_rate = svd_rate

    calib = get_calib_data(tokenizer, nsamples=NSAMPLES, seed=SEED)
    inps, kwargs = capture_layer0_inputs(model, calib)
    for li, layer in enumerate(model.model.layers):
        mods = dict(layer.named_modules())
        for group in SEQUENTIAL:
            names = [full_name(s) for s in group]
            for n in names:
                if n in OTHER_QUANT:
                    q, m = OTHER_QUANT[n], mods[n]
                    m.weight.data = rtn_quantize_weight(m.weight, q["wbits"], q["group_size"], q["sym"])
                    set_compressed_bits(m, quant_storage_bits(m.out_features, m.in_features,
                                                              q["wbits"], q["group_size"], q["sym"]))
            svd_names = [n for n in names if n in SVD_TARGETS]
            if not svd_names:
                continue
            Hs = collect_hessians(layer, svd_names, inps, kwargs)
            for n in svd_names:
                m = mods[n]
                B, A = whitened_svd(m.weight, Hs[n], rank)
                q = FACTOR_QUANT
                B = rtn_quantize_weight(B, q["wbits"], q["group_size"], q["sym"])
                A = rtn_quantize_weight(A, q["wbits"], q["group_size"], q["sym"])
                new = LowRankLinear(B, A)
                set_compressed_bits(new, _lowrank_bits(m.out_features, m.in_features, rank))
                parent, child = n.split(".")
                setattr(getattr(layer, parent), child, new)
            del Hs
            mods = dict(layer.named_modules())
        inps = run_layer(layer, inps, kwargs)
        log(f"[case3] layer {li} done")
    torch.cuda.empty_cache()
