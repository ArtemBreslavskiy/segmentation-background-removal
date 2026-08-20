from pydantic import BaseModel, Field, model_validator
from typing import Literal, Union


class BaseLossConfig(BaseModel):
    model_config = {"extra": "forbid"}
    type: str


class MaskedFocalLossConfig(BaseLossConfig):
    type: Literal["masked_focal_loss"]
    alpha: float = Field(ge=0)
    gamma: float = Field(ge=0)
    reduction: str = "mean"
    smooth: float = Field(1e-6, ge=0)


class MaskedTverskyLossConfig(BaseLossConfig):
    type: Literal["masked_tversky_loss"]
    alpha: float = Field(ge=0)
    beta: float = Field(ge=0)
    smooth: float = Field(1e-6, ge=0)


class ComboLossConfig(BaseLossConfig):
    type: Literal["combo_loss"]
    weights: list[float]
    loss_functions: list[Union[
        MaskedFocalLossConfig,
        MaskedTverskyLossConfig
    ]] = Field(discriminator="type")

    @model_validator(mode="after")
    def validate_weights_and_functions(self) -> "ComboLossConfig":
        if any(w < 0 for w in self.weights):
            raise ValueError("All weights must be >= 0")
        if len(self.weights) != len(self.loss_functions):
            raise ValueError(
                f"Number of weights ({len(self.weights)}) does not match "
                f"number of loss functions ({len(self.loss_functions)})"
            )
        return self
