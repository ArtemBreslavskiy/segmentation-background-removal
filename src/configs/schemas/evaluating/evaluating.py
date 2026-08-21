from pydantic import BaseModel, Field
from typing import Union
import src.configs.schemas.evaluating.metric as metric


class EvaluatingConfig(BaseModel):
    use_cuda: bool
    metrics: list[Union[
        metric.IOUConfig,
        metric.AccuracyConfig,
        metric.PrecisionConfig,
        metric.RecallConfig,
        metric.F1Config,
        metric.SpecificityConfig,
        metric.MCCConfig,
    ]] = Field(discriminator="name")
