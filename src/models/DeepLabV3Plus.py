import segmentation_models_pytorch as smp
import torch.nn as nn


class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        encoder_name: str = "mit_b5",
        pretrained: bool = True,
        num_classes: int = 1,
        use_aux: bool = False,
        use_gradient_checkpointing: bool = False
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
            self.model.encoder.set_grad_checkpointing(True)

    def forward(self, x):
        x = self.model(x)

        if isinstance(x, dict):
            if self.training and self.use_aux:
                return x["out"], x["aux"]
            else:
                return x["out"]
        return x
