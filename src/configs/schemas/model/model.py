from pydantic import BaseModel, Field
from typing import Union
import src.configs.schemas.model.init as init


class ModelConfig(BaseModel):
    model_name: str
    init: Union[init.DeepLabV3PlusConfig, init.SegFormerConfig] = Field(discriminator="type")
