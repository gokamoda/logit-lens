from dataclasses import dataclass
from typing import Any

import torch

from .hook import PostAttnHookForHiddenStates, PostHook, PostLayerHookForHiddenStates
from .typing import BATCH, HIDDEN_DIM, LAYER, SEQUENCE, Tensor


@dataclass
class LogitLensResult:
    post_emb_token_ids: Tensor[SEQUENCE, HIDDEN_DIM] = None
    post_emb_probs: Tensor[SEQUENCE, HIDDEN_DIM] = None
    post_attn_token_ids: Tensor[LAYER, SEQUENCE, HIDDEN_DIM] = None
    post_attn_probs: Tensor[LAYER, SEQUENCE, HIDDEN_DIM] = None
    post_layer_token_ids: Tensor[LAYER, SEQUENCE, HIDDEN_DIM] = None
    post_layer_probs: Tensor[LAYER, SEQUENCE, HIDDEN_DIM] = None


@dataclass
class BatchLogitLensResult:
    post_emb_token_ids: Tensor[BATCH, SEQUENCE, HIDDEN_DIM] = None
    post_emb_probs: Tensor[BATCH, SEQUENCE, HIDDEN_DIM] = None
    post_attn_token_ids: Tensor[BATCH, LAYER, SEQUENCE, HIDDEN_DIM] = None
    post_attn_probs: Tensor[BATCH, LAYER, SEQUENCE, HIDDEN_DIM] = None
    post_layer_token_ids: Tensor[BATCH, LAYER, SEQUENCE, HIDDEN_DIM] = None
    post_layer_probs: Tensor[BATCH, LAYER, SEQUENCE, HIDDEN_DIM] = None

    # get by [batch_index]
    def __getitem__(self, index: int) -> "LogitLensResult":
        if index > 0:
            raise Warning("LogitLens currently do not support padding")
        return LogitLensResult(
            post_emb_token_ids=self.post_emb_token_ids[index]
            if self.post_emb_token_ids is not None
            else None,
            post_emb_probs=self.post_emb_probs[index]
            if self.post_emb_probs is not None
            else None,
            post_attn_token_ids=self.post_attn_token_ids[index]
            if self.post_attn_token_ids is not None
            else None,
            post_attn_probs=self.post_attn_probs[index]
            if self.post_attn_probs is not None
            else None,
            post_layer_token_ids=self.post_layer_token_ids[index]
            if self.post_layer_token_ids is not None
            else None,
            post_layer_probs=self.post_layer_probs[index]
            if self.post_layer_probs is not None
            else None,
        )


class LogitLens:
    """
    A class representing a Logit Lens for analyzing model logits.

    Basic usage:
        model = ...  # Your model here

        with LogitLens(model, post_layer=True) as ll:
            outputs = model(input_ids)
            ll_result = ll.get_logits()
    """

    model: Any
    topn: int
    post_emb: bool
    post_emb_hook: PostHook | None
    post_attn: bool
    post_attn_hooks: list[PostAttnHookForHiddenStates]
    post_layer: bool
    post_layer_hooks: list[PostLayerHookForHiddenStates]

    def __init__(
        self,
        model,
        post_emb: bool = False,
        post_attn: bool = False,
        post_layer: bool = False,
        topn: int = 5,
    ):
        self.model = model

        self.post_emb = post_emb
        self.post_emb_hook = None

        self.post_attn = post_attn
        self.post_attn_hooks = []

        self.post_layer = post_layer
        self.post_layer_hooks = []

        self.topn = topn

    def lens_forward(
        self,
        x: torch.Tensor,
    ):
        logits = torch.nn.functional.softmax(
            self.model.lm_head(self.model.model.norm(x)), dim=-1
        )
        probs, indices = torch.topk(logits, self.topn, dim=-1)
        return probs, indices

    def __enter__(self):
        if self.post_emb:
            self.post_emb_hook = PostHook(self.model.model.embed_tokens)

        if self.post_attn:
            for layer in self.model.model.layers:
                self.post_attn_hooks.append(
                    PostAttnHookForHiddenStates(layer.self_attn)
                )

        if self.post_layer:
            for module in self.model.model.layers:
                self.post_layer_hooks.append(PostLayerHookForHiddenStates(module))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.post_emb and self.post_emb_hook is not None:
            self.post_emb_hook.remove()

        if self.post_attn:
            for hook in self.post_attn_hooks:
                hook.remove()

        if self.post_layer:
            for hook in self.post_layer_hooks:
                hook.remove()

    def get_results(self):
        result = BatchLogitLensResult()

        if (
            self.post_emb
            and self.post_emb_hook is not None
            and self.post_emb_hook.result is not None
        ):
            post_emb_probs, post_emb_token_ids = self.lens_forward(
                self.post_emb_hook.result
            )
            result.post_emb_probs = post_emb_probs
            result.post_emb_token_ids = post_emb_token_ids

        if self.post_attn:
            post_attn_probs, post_attn_token_ids = self.lens_forward(
                torch.stack([hook.result for hook in self.post_attn_hooks], dim=1)
            )
            result.post_attn_probs = post_attn_probs
            result.post_attn_token_ids = post_attn_token_ids

        if self.post_layer:
            post_layer_probs, post_layer_token_ids = self.lens_forward(
                torch.stack([hook.result for hook in self.post_layer_hooks], dim=1)
            )
            result.post_layer_probs = post_layer_probs
            result.post_layer_token_ids = post_layer_token_ids

        return result
