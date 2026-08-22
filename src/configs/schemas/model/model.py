from pydantic import BaseModel, Field
from typing import Union
import src.configs.schemas.model.model_init as model_init


class ModelConfig(BaseModel):
    model_name: str
    init: Union[model_init.DeepLabV3PlusConfig, model_init.SegFormerConfig] = Field(discriminator="type")
