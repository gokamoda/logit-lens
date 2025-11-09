from .gpt2 import GPT2LMHEADMODEL_ALIAS, extract_gpt2_hidden_states

ARCHITECTURE_ALIAS = {"GPT2LMHeadModel": GPT2LMHEADMODEL_ALIAS}
HIDDEN_STATE_EXTRACTORS = {
    "GPT2Block": extract_gpt2_hidden_states,
}
