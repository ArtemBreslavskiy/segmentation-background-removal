import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean", smooth=1e-6):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction
        self.smooth = float(smooth)

    def forward(self, pred, target, valid_mask=None):
        prob = torch.sigmoid(pred)

        prob = torch.clamp(prob, self.smooth, 1 - self.smooth)
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t = prob * target + (1 - prob) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal_loss = alpha_weight * focal_weight * ce_loss

        if valid_mask is not None:
            focal_loss = focal_loss * valid_mask
            total_valid = valid_mask.sum()
            if self.reduction == "mean":
                if total_valid > 0:
                    return focal_loss.sum() / total_valid
                else:
                    return focal_loss.sum()
            elif self.reduction == "sum":
                return focal_loss.sum()
            else:
                return focal_loss
        else:
            if self.reduction == "mean":
                return focal_loss.mean()
            elif self.reduction == "sum":
                return focal_loss.sum()
            else:
                return focal_loss
