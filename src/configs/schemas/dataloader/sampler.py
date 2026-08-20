from pydantic import BaseModel, Field
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
