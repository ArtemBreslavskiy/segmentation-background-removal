import torch
import torch.nn as nn


class MaskedPixelwiseSoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, valid_mask=None):
        B = logits.size(0)
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(B, -1)
        targets_flat = targets.view(B, -1)
        if valid_mask is not None:
            valid_flat = valid_mask.view(B, -1)
            probs_flat = probs_flat * valid_flat
            targets_flat = targets_flat * valid_flat
            pixels_per_image = valid_flat.sum(1)
            total_pixels = pixels_per_image.sum()
        else:
            num_pixels_per_image = targets_flat.shape[1]
            pixels_per_image = torch.full((B,), num_pixels_per_image, device=logits.device)
            total_pixels = B * num_pixels_per_image

        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss_per_image = 1.0 - dice_score

        loss = (dice_loss_per_image * pixels_per_image).sum() / total_pixels
        return loss
