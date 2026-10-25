"""Shared utilities for Part 2. Every compressed target Linear carries `_compressed_bits`
(weights + scales/zeros + low-rank factors), compared by `evaluate.compression_rate`
against the FP16 size stored by `record_base`."""
import math
import random

import torch
import torch.nn as nn

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SEQUENTIAL = [["q_proj", "k_proj", "v_proj"], ["o_proj"], ["gate_proj", "up_proj"], ["down_proj"]]
FP16_BITS = 16


def log(*args):
    import sys
    print(*args, file=sys.stderr, flush=True)


def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def submodule_name(name):
    return ".".join(name.split(".")[-2:])


def iter_targets(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in TARGETS and ".layers." in name:
            yield name, module


def record_base(model):
    numel = {}
    for name, m in iter_targets(model):
        numel[name] = m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
    model._base_linear_numel = numel
    model._base_linear_bits = sum(numel.values()) * FP16_BITS
    return model._base_linear_bits


def set_compressed_bits(module, bits):
    module._compressed_bits = int(bits)


def quant_storage_bits(out_features, in_features, wbits, group_size=-1, sym=False, scale_bits=FP16_BITS):
    """group_size=-1 means one group per row; each group stores a scale and, if
    asymmetric, a wbits-bit zero point."""
    gs = in_features if group_size <= 0 else group_size
    n_groups = out_features * math.ceil(in_features / gs)
    per_group = scale_bits + (0 if sym else wbits)
    return out_features * in_features * wbits + n_groups * per_group


def module_bits(module):
    if hasattr(module, "_compressed_bits"):
        return module._compressed_bits
    return sum(p.numel() for p in module.parameters()) * FP16_BITS


def _range_to_params(xmin, xmax, maxq, sym):
    scale = ((xmax - xmin) / maxq).clamp(min=1e-8).half().float()  # stored as fp16
    if sym:
        zero = torch.full_like(scale, (maxq + 1) / 2)
    else:
        zero = torch.round(-xmin / scale).clamp(0, maxq)
    return scale, zero


def quant_params(w, wbits, sym, mse=False, grid=100, maxshrink=0.8, norm=2.4):
    """Per-row scale/zero; mse=True shrinks the range to minimise sum |w - Q(w)|^norm."""
    maxq = 2 ** wbits - 1
    xmin = w.amin(dim=1, keepdim=True).clamp(max=0)
    xmax = w.amax(dim=1, keepdim=True).clamp(min=0)
    if sym:
        xmax = torch.maximum(xmin.abs(), xmax)
        xmin = -xmax
    scale, zero = _range_to_params(xmin, xmax, maxq, sym)
    if not mse:
        return scale, zero
    best = torch.full_like(scale, float("inf"))
    for i in range(int(maxshrink * grid)):
        p = 1 - i / grid
        s, z = _range_to_params(p * xmin, p * xmax, maxq, sym)
        err = (quantize(w, s, z, wbits) - w).abs().pow(norm).sum(dim=1, keepdim=True)
        better = err < best
        best = torch.where(better, err, best)
        scale = torch.where(better, s, scale)
        zero = torch.where(better, z, zero)
    return scale, zero


def quantize(x, scale, zero, wbits):
    maxq = 2 ** wbits - 1
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


def rtn_quantize_weight(w, wbits, group_size=-1, sym=False):
    out_f, in_f = w.shape
    gs = in_f if group_size <= 0 else group_size
    wf = w.float().reshape(out_f, in_f // gs, gs).reshape(-1, gs)
    scale, zero = quant_params(wf, wbits, sym)
    return quantize(wf, scale, zero, wbits).reshape(out_f, in_f).to(w.dtype)


def get_calib_data(tokenizer, nsamples=128, seqlen=2048, seed=0):
    """Random windows from WikiText-2 train; the test split is never used."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    gen = random.Random(seed)
    samples = []
    for _ in range(nsamples):
        i = gen.randint(0, ids.shape[1] - seqlen - 1)
        samples.append(ids[:, i:i + seqlen])
    return torch.cat(samples, dim=0)


class _Catcher(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.attention_type = getattr(layer, "attention_type", None)  # read by Qwen3Model.forward
        self.inps, self.kwargs = [], None

    def forward(self, hidden_states, **kwargs):
        self.inps.append(hidden_states)
        self.kwargs = kwargs
        raise ValueError


@torch.no_grad()
def capture_layer0_inputs(model, calib):
    layers = model.model.layers
    catcher = _Catcher(layers[0])
    layers[0] = catcher
    device = next(model.parameters()).device
    for i in range(calib.shape[0]):
        try:
            model(calib[i:i + 1].to(device))
        except ValueError:
            pass
    layers[0] = catcher.layer
    kwargs = {k: v for k, v in catcher.kwargs.items() if k not in ("past_key_values", "past_key_value", "use_cache")}
    return torch.cat(catcher.inps, dim=0), kwargs


@torch.no_grad()
def run_layer(layer, inps, kwargs):
    outs = torch.empty_like(inps)
    for i in range(inps.shape[0]):
        out = layer(inps[i:i + 1], **kwargs)
        outs[i:i + 1] = out[0] if isinstance(out, tuple) else out
    return outs


class HessianCollector:
    """H = 2/N * sum(x x^T) over the inputs of a Linear."""

    def __init__(self, module):
        cols = module.in_features
        self.H = torch.zeros(cols, cols, device=module.weight.device, dtype=torch.float32)
        self.n = 0
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        x = inp[0].reshape(-1, inp[0].shape[-1])
        t = x.shape[0]
        self.H *= self.n / (self.n + t)
        self.n += t
        # fp16 GEMM for speed on T4; per-channel scaling keeps outlier channels from
        # overflowing or pushing the others into fp16 underflow.
        c = x.abs().amax(dim=0).float().clamp(min=1e-6)
        xh = (x.float() / c).half()
        self.H += (xh.t() @ xh).float() * (c.unsqueeze(1) * c.unsqueeze(0)) * (2 / self.n)

    def remove(self):
        self.handle.remove()


class _StopForward(Exception):
    pass


@torch.no_grad()
def collect_hessians(layer, names, inps, kwargs):
    """Each forward stops once every requested module has seen its input."""
    mods = dict(layer.named_modules())
    collectors = {n: HessianCollector(mods[n]) for n in names}
    seen = set()

    def stop_when_done(name):
        def hook(module, inp, out):
            seen.add(name)
            if len(seen) == len(names):
                raise _StopForward
        return mods[name].register_forward_hook(hook)

    stoppers = [stop_when_done(n) for n in names]  # registered after the collectors, so they run last
    for i in range(inps.shape[0]):
        seen.clear()
        try:
            layer(inps[i:i + 1], **kwargs)
        except _StopForward:
            pass
    for h in stoppers:
        h.remove()
    for c in collectors.values():
        c.remove()
    return {n: c.H for n, c in collectors.items()}


@torch.no_grad()
def gptq_quantize(weight, H, wbits, group_size=-1, sym=False, actorder=True, percdamp=0.01, blocksize=128,
                  mse=False):
    """GPTQ with act-order and static groups (scales from the original column order),
    so no per-column group index needs to be stored."""
    W = weight.detach().float().clone()
    cols = W.shape[1]
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    gs = cols if group_size <= 0 else group_size
    groups = [quant_params(W[:, g:g + gs], wbits, sym, mse) for g in range(0, cols, gs)]

    perm = torch.argsort(torch.diag(H), descending=True) if actorder else torch.arange(cols, device=W.device)
    W, H = W[:, perm], H[perm][:, perm]
    invperm = torch.argsort(perm)

    damp = percdamp * torch.mean(torch.diag(H))
    H += damp * torch.eye(cols, device=W.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)

    gidx = (perm // gs).tolist()  # on the host to avoid a GPU sync per column
    Q = torch.zeros_like(W)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w, d = W1[:, i], Hinv1[i, i]
            scale, zero = groups[gidx[i1 + i]]
            q = quantize(w.unsqueeze(1), scale, zero, wbits).flatten()
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q[:, invperm].to(weight.dtype)


@torch.no_grad()
def gptq_model(model, calib, config_fn, actorder=True, percdamp=0.01):
    """config_fn(layer_idx, name) -> quant cfg, or None to keep the module in fp16."""
    inps, kwargs = capture_layer0_inputs(model, calib)
    for li, layer in enumerate(model.model.layers):
        for names in SEQUENTIAL:
            todo = [full_name(n) for n in names if config_fn(li, full_name(n)) is not None]
            if not todo:
                continue
            Hs = collect_hessians(layer, todo, inps, kwargs)
            mods = dict(layer.named_modules())
            for n in todo:
                gptq_module(mods[n], Hs[n], config_fn(li, n), actorder, percdamp)
            del Hs
        inps = run_layer(layer, inps, kwargs)
        log(f"[gptq] layer {li} done")
    torch.cuda.empty_cache()


@torch.no_grad()
def gptq_module(m, H, cfg, actorder=True, percdamp=0.01):
    gs, sym = cfg.get("group_size", -1), cfg.get("sym", False)
    m.weight.data = gptq_quantize(m.weight, H, cfg["wbits"], gs, sym, actorder, percdamp, mse=cfg.get("mse", False))
    set_compressed_bits(m, quant_storage_bits(m.out_features, m.in_features, cfg["wbits"], gs, sym))


def make_config_fn(default, rules=()):
    """rules: [(first_layer, last_layer, name_substring, cfg_or_None)]; later rules win."""
    def config_fn(li, name):
        cfg = default
        for lo, hi, pat, c in rules:
            if lo <= li <= hi and pat in name:
                cfg = c
        return cfg
    return config_fn


def gptq_case(model, tokenizer, default, rules=(), nsamples=128, seed=0):
    seed_everything(seed)
    record_base(model)
    calib = get_calib_data(tokenizer, nsamples=nsamples, seed=seed)
    gptq_model(model, calib, make_config_fn(default, rules))


class LowRankLinear(nn.Module):
    """y = A(B(x)); replaces one target nn.Linear."""

    def __init__(self, B, A):
        super().__init__()
        rank, in_f = B.shape
        out_f = A.shape[0]
        self.in_features, self.out_features, self.rank = in_f, out_f, rank
        self.B = nn.Linear(in_f, rank, bias=False, device=B.device, dtype=B.dtype)
        self.A = nn.Linear(rank, out_f, bias=False, device=A.device, dtype=A.dtype)
        self.B.weight.data.copy_(B)
        self.A.weight.data.copy_(A)

    def forward(self, x):
        return self.A(self.B(x))


@torch.no_grad()
def whitened_svd(weight, H, rank, percdamp=0.01):
    """SVD-LLM: truncate W S (H = S S^T) so the dropped directions minimise the output
    error. Returns B [rank, in], A [out, rank] with W ~= A @ B."""
    W = weight.detach().float()
    H = H.double()
    eye = torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
    damp = percdamp * torch.mean(torch.diag(H))
    while True:  # grow the damping until the Cholesky succeeds
        S, info = torch.linalg.cholesky_ex(H + damp * eye)
        if info.item() == 0:
            break
        damp *= 10
    S = S.float()
    U, sig, Vh = torch.linalg.svd(W @ S, full_matrices=False)
    root = sig[:rank].sqrt()
    A = U[:, :rank] * root
    B = torch.linalg.solve_triangular(S, root.unsqueeze(1) * Vh[:rank], upper=False, left=False)
    return B.to(weight.dtype), A.to(weight.dtype)


def lowrank_numel(out_features, in_features, rank):
    return rank * (out_features + in_features)


def full_name(short):
    return ("self_attn." if short in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp.") + short
