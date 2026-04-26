import segmentation_models_pytorch as smp
import torch.nn as nn


class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        encoder_name: str = "mit_b5",
        pretrained: bool = True,
        num_classes: int = 1,
        use_aux: bool = False,
        use_gradient_checkpointing: bool = False,
        group_norm_groups: int = 0,
        group_norm_eps: float = 1e-5,
        group_norm_preserve_weights: bool = True,
    ):
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if group_norm_groups < 0:
            raise ValueError("group_norm_groups must be >= 0")

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

        if group_norm_groups >= 1:
            self._replace_bn_with_gn(
                self.model,
                group_norm_groups,
                group_norm_eps,
                group_norm_preserve_weights,
            )

        if use_gradient_checkpointing:
            if hasattr(self.model.encoder, "set_grad_checkpointing"):
                self.model.encoder.set_grad_checkpointing(True)
            elif hasattr(self.model.encoder, "gradient_checkpointing"):
                self.model.encoder.gradient_checkpointing = True
            else:
                print(
                    "Warning: Gradient checkpointing could not be enabled for this encoder."
                )

    def _replace_bn_with_gn(
        self,
        module,
        num_groups: int = 32,
        eps: float = 1e-5,
        preserve_weights: bool = True,
    ):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_features = child.num_features
                actual_groups = min(num_features, num_groups) if num_features > 0 else 1
                while num_features % actual_groups != 0 and actual_groups > 1:
                    actual_groups -= 1

                gn = nn.GroupNorm(actual_groups, num_features, eps=eps)
                if preserve_weights:
                    gn.weight.data.copy_(child.weight.data)
                    gn.bias.data.copy_(child.bias.data)
                setattr(module, name, gn)
            else:
                self._replace_bn_with_gn(child, num_groups, eps, preserve_weights)

    def forward(self, x):
        x = self.model(x)

        if isinstance(x, dict):
            if self.training and self.use_aux:
                return x["out"], x["aux"]
            else:
                return x["out"]
        return x
