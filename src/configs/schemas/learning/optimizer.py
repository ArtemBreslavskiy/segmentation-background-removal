from pydantic import BaseModel, Field
from typing import Literal


class BaseOptimizerConfig(BaseModel):
    model_config = {"extra": "forbid"}
    type: str


class TorchAdamWConfig(BaseOptimizerConfig):
    type: Literal["torch_adam_w"]
    lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    fused: bool
