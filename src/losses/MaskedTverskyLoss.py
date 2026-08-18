import torch
import torch.nn as nn


class MaskedTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)

    def forward(self, logits, targets, valid_mask=None):
        B = logits.size(0)
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(B, -1)

        targets_flat = targets.view(B, -1).float()
        if targets_flat.max() > 1.0:
            targets_flat = targets_flat / 255.0

        if valid_mask is not None:
            valid_flat = valid_mask.view(B, -1).float()
            probs_flat = probs_flat * valid_flat
            targets_flat = targets_flat * valid_flat
            pixels_per_image = valid_flat.sum(dim=1)
            total_pixels = pixels_per_image.sum()
        else:
            num_pixels = targets_flat.size(1)
            pixels_per_image = torch.full((B,), num_pixels, device=logits.device)
            total_pixels = B * num_pixels

        if total_pixels == 0:
            return torch.tensor(0.0, device=logits.device)

        TP = (probs_flat * targets_flat).sum(dim=1)
        FP = (probs_flat * (1 - targets_flat)).sum(dim=1)
        FN = ((1 - probs_flat) * targets_flat).sum(dim=1)

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        loss_per_image = 1.0 - tversky

        loss = (loss_per_image * pixels_per_image).sum() / total_pixels
        return loss
