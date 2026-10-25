"""Case 2: INT3 with groups of 512 columns (3.037 bits/weight -> 81.02%)."""
from common import gptq_case

DEFAULT = dict(wbits=3, group_size=512, sym=False, mse=True)
RULES = []
NSAMPLES = 128
SEED = 0


def compress(model, tokenizer, rate=None, wbits=None):
    gptq_case(model, tokenizer, DEFAULT, RULES, nsamples=NSAMPLES, seed=SEED)
