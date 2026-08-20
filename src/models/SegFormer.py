import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig


class SegFormer(nn.Module):
    def __init__(
        self,
        encoder_name: str = "nvidia/mit-b5",
        pretrained: bool = True,
        num_classes: int = 1,
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

        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                encoder_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig.from_pretrained(encoder_name)
            config.num_labels = num_classes
            self.model = SegformerForSemanticSegmentation(config)

        if group_norm_groups >= 1:
            self._replace_bn_with_gn(
                self.model,
                group_norm_groups,
                group_norm_eps,
                group_norm_preserve_weights,
            )

        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

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
                if preserve_weights and hasattr(child, "weight"):
                    gn.weight.data.copy_(child.weight.data)
                    gn.bias.data.copy_(child.bias.data)
                setattr(module, name, gn)
            else:
                self._replace_bn_with_gn(child, num_groups, eps, preserve_weights)

    def forward(self, x):
        outputs = self.model(x)
        logits = outputs.logits
        H, W = x.shape[2], x.shape[3]
        logits = F.interpolate(
            logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )
        return logits
