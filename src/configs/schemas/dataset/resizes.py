from pydantic import BaseModel, Field
from typing import Union
import src.configs.schemas.dataset.resize as mode


class ResizesConfig(BaseModel):
    train: Union[
        mode.DefaultResizeConfig,
        mode.CropConfig,
        mode.ResizeMixAConfig,
        mode.ResizeMixBConfig,
    ] = Field(description="mode")
    test: Union[
        mode.DefaultResizeConfig,
        mode.CropConfig,
        mode.ResizeMixAConfig,
        mode.ResizeMixBConfig,
    ] = Field(description="mode")
    val: Union[
        mode.DefaultResizeConfig,
        mode.CropConfig,
        mode.ResizeMixAConfig,
        mode.ResizeMixBConfig,
    ] = Field(description="mode")
