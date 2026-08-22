from pydantic import BaseModel, Field
from typing import Literal


class BaseResizeModeConfig(BaseModel):
    model_config = {"extra": "forbid"}
    mode: str


class DefaultResizeConfig(BaseResizeModeConfig):
    mode: Literal["resize"]


class CropConfig(BaseResizeModeConfig):
    mode: Literal["crop"]
    min_foreground_share: float = Field(0.0, ge=0)


class ResizeMixAConfig(BaseResizeModeConfig):
    mode: Literal["mix-a"]
    area_threshold_mix: int = Field(gt=0)
    min_foreground_share: float = Field(0.0, ge=0)


class ResizeMixBConfig(BaseResizeModeConfig):
    mode: Literal["mix-b"]
    area_threshold_mix: int = Field(gt=0)
    min_foreground_share: float = Field(0.0, ge=0)
