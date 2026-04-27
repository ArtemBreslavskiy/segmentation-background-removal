import torch.nn as nn


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        B = targets.size(0)
        probs = nn.functional.sigmoid(logits)

        probs = probs.view(B, -1)
        targets = targets.view(B, -1)

        intersection = (probs * targets).sum(dim=1)
        cardinality = probs.sum(dim=1) + targets.sum(dim=1)

        score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        return 1 - score.mean()
