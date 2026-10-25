"""Case 1: INT4 g128 everywhere, gate_proj (the least sensitive projection) INT3 g128."""
from common import gptq_case

INT4 = dict(wbits=4, group_size=128, sym=False, mse=True)
INT3 = dict(wbits=3, group_size=128, sym=False, mse=True)
DEFAULT = INT4
RULES = [(0, 35, "mlp.gate_proj", INT3)]  # (first_layer, last_layer, module, cfg)
NSAMPLES = 128
SEED = 0


def compress(model, tokenizer, rate=None, wbits=None):
    gptq_case(model, tokenizer, DEFAULT, RULES, nsamples=NSAMPLES, seed=SEED)
