from dataclasses import dataclass
from typing import Any

import torch

from .hook import PostHook
from .typing import BATCH, HIDDEN_DIM, LAYER, SEQUENCE, Tensor


@dataclass
class LogitLensResult:
    post_layer_token_ids: Tensor[LAYER, SEQUENCE, HIDDEN_DIM] = None
    post_layer_probs: Tensor[LAYER, SEQUENCE, HIDDEN_DIM] = None


@dataclass
class BatchLogitLensResult:
    post_layer_token_ids: Tensor[BATCH, LAYER, SEQUENCE, HIDDEN_DIM] = None
    post_layer_probs: Tensor[BATCH, LAYER, SEQUENCE, HIDDEN_DIM] = None

    # get by [batch_index]
    def __getitem__(self, index: int) -> "LogitLensResult":
        if index > 0:
            raise Warning("LogitLens currently do not support padding")
        return LogitLensResult(
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
    post_layer: bool
    post_layer_hooks: list[PostHook]

    def __init__(self, model, post_layer: bool = True, topn: int = 5):
        self.model = model
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
        if self.post_layer:
            for module in self.model.model.layers:
                self.post_layer_hooks.append(PostHook(module))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self.post_layer_hooks:
            hook.remove()

    def get_results(self):
        result = BatchLogitLensResult()

        if self.post_layer:
            post_layer_probs, post_layer_token_ids = self.lens_forward(
                torch.stack([hook.result for hook in self.post_layer_hooks], dim=1)
            )
            result.post_layer_probs = post_layer_probs
            result.post_layer_token_ids = post_layer_token_ids

        return result
