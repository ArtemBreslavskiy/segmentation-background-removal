import segmentation_models_pytorch as smp
import torch.nn as nn
from typing import Dict, Any


class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        encoder_name: str = "mit_b5",
        pretrained: bool = True,
        num_classes: int = 1,
        use_aux: bool = False,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.use_aux = use_aux

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            encoder_output_stride=16,
            classes=num_classes,
            activation=None,
            aux_params=(
                {
                    "pooling": "avg",
                    "dropout": 0.1,
                    "activation": None,
                    "classes": num_classes,
                }
                if use_aux
                else None
            ),
        )

        if use_gradient_checkpointing:
            if hasattr(self.model.encoder, 'set_grad_checkpointing'):
                self.model.encoder.set_grad_checkpointing(True)
            elif hasattr(self.model.encoder, 'gradient_checkpointing'):
                self.model.encoder.gradient_checkpointing = True
            else:
                print("Warning: Gradient checkpointing could not be enabled for this encoder.")

    def forward(self, x):
        x = self.model(x)

        if isinstance(x, dict):
            if self.training and self.use_aux:
                return x["out"], x["aux"]
            else:
                return x["out"]
        return x


def _convert_value(value: Any):
    if isinstance(value, dict):
        return {k: _convert_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_convert_value(item) for item in value]
    if isinstance(value, str):
        try:
            if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                return int(value)

            float_val = float(value)
            if float_val.is_integer() and "." not in value:
                return int(float_val)
            return float_val
        except ValueError:
            return value
    return value


def create_model(config: Dict):
    return DeepLabV3Plus(**_convert_value(config["model"]["params"]))