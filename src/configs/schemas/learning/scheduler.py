from pydantic import BaseModel, Field
from typing import Literal


class BaseSchedulerConfig(BaseModel):
    model_config = {"extra": "forbid"}
    type: str


class TorchCosineAnnealingWarmRestartsConfig(BaseSchedulerConfig):
    type: Literal["torch_cosine_annealing_warm_restarts"]
    T_0: int = Field(gt=0)
    T_mult: float = Field(gt=0)
    eta_min: float = Field(ge=0)
