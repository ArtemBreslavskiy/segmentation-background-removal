from pydantic import BaseModel, Field
from typing import Literal, Union
import src.configs.schemas.learning.loss as loss
import src.configs.schemas.learning.optimizer as optimizer
import src.configs.schemas.learning.scheduler as scheduler


class LearningConfig(BaseModel):
    use_cuda: bool
    use_fp16: bool = False
    max_clip_grad_norm: float = Field(1.0, gt=0)
    threshold: float = Field(ge=0, le=0)
    epochs: int = Field(gt=0)
    early_stopping_patience: int = Field(ge=0)
    log_interval: int = Field(ge=0)
    save_criterion: str
    mode: Literal["max", "min"]
    compile_model: bool = False
    compile_dynamic: bool = False
    compile_options: str = "default"
    accumulation_steps: int = Field(1, gt=0)
    pixels_per_step: int = Field(0, ge=0)
    loss: Union[
        loss.MaskedFocalLossConfig,
        loss.MaskedTverskyLossConfig,
        loss.ComboLossConfig,
    ]
    optimizer: Union[
        optimizer.TorchAdamWConfig,
    ]
    scheduler: Union[
        scheduler.TorchCosineAnnealingWarmRestartsConfig
    ]
