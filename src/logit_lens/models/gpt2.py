import torch

GPT2LMHEADMODEL_ALIAS = [
    ("model", "transformer"),
    ("model.embed_tokens", "transformer.wte"),
    ("model.lm_head", "transformer.lm_head"),
    ("model.layers", "transformer.h"),
    ("model.layers[*].self_attn", "transformer.h[*].attn"),
    ("model.norm", "transformer.ln_f"),
]


def extract_gpt2_hidden_states(layer_output: tuple) -> torch.Tensor:
    return layer_output[0]
