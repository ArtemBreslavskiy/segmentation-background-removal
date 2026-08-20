from pydantic import BaseModel, Field
from typing import Literal


class BaseModelInitConfig(BaseModel):
    model_config = {"extra": "forbid"}
    type: str


class DeepLabV3PlusConfig(BaseModelInitConfig):
    type: Literal["DeepLabV3Plus"]
    encoder_name: str
    num_classes: int = Field(gt=0)
    pretrained: bool = True
    use_aux: bool = False
    use_gradient_checkpointing: bool = False
    group_norm_groups: int = Field(0, ge=0)
    group_norm_eps: float = Field(1e-5, gt=0)
    group_norm_preserve_weights: bool = True


class SegFormerConfig(BaseModelInitConfig):
    type: Literal["SegFormer"]
    encoder_name: str
    num_classes: int = Field(gt=0)
    pretrained: bool = True
    use_gradient_checkpointing: bool = False
    group_norm_groups: int = Field(0, ge=0)
    group_norm_eps: float = Field(1e-5, gt=0)
    group_norm_preserve_weights: bool = True
