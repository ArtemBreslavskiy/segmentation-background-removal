from pydantic import BaseModel, Field, field_validator
from typing import Literal


class BaseSamplerConfig(BaseModel):
    model_config = {"extra": "forbid"}
    type: str


class WeightedDynamicBucketBatchSamplerConfig(BaseSamplerConfig):
    type: Literal["wdb"]
    max_batch_size: int = Field(gt=0)
    min_batch_size: int = Field(gt=0)
    max_load: int = Field(gt=0)
    replacement: bool
    skip_overload_examples: bool
    send_overload_report: bool
    weights: dict[str, float] = Field(default_factory=dict)

    @field_validator("weights")
    def validate_weights(cls, v: dict[str, float]) -> dict[str, float]:
        if v is None:
            return {}
        for name, weight in v.items():
            if weight <= 0:
                raise ValueError(f"Weight for '{name}' must be > 0, got {weight}")
        return v
