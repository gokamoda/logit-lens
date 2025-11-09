from typing import Annotated

import torch
from typing_extensions import Generic, TypeVarTuple

T = TypeVarTuple("T")


class Tensor(Generic[*T], torch.Tensor):  # type: ignore
    pass


BATCH = Annotated[int, "batch"]
LAYER = Annotated[int, "layer"]
SEQUENCE = Annotated[int, "sequence"]
HEAD = Annotated[int, "head"]
HIDDEN_DIM = Annotated[int, "hidden_dim"]
HEAD_DIM = Annotated[int, "head_dim"]
