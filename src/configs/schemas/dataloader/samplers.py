from pydantic import BaseModel, Field
from typing import Union
import src.configs.schemas.dataloader.sampler as sampler


class SamplersConfig:
    train: Union[sampler.WeightedDynamicBucketBatchSamplerConfig] = Field(description="type")
    test: Union[sampler.WeightedDynamicBucketBatchSamplerConfig] = Field(description="type")
    val: Union[sampler.WeightedDynamicBucketBatchSamplerConfig] = Field(description="type")
