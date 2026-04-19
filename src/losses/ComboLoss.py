import torch.nn as nn


class ComboLoss(nn.Module):
    def __init__(self, loss_functions: list[nn.Module], weights: list[float] = None):
        super().__init__()
        if not loss_functions:
            raise ValueError("loss_functions cannot be empty")

        self.loss_functions = nn.ModuleList(loss_functions)
        self.loss_functions_names = []
        name_counts = {}
        for lf in loss_functions:
            base = type(lf).__name__
            count = name_counts.get(base, 0)
            name_counts[base] = count + 1
            if count > 0:
                name = f"{base}_{count}"
            else:
                name = base
            self.loss_functions_names.append(name)

        if weights is None:
            weights = [1.0] * len(loss_functions)

        if len(loss_functions) != len(weights):
            raise ValueError("Number of weights must match number of loss functions")

        total_weights = sum(weights)

        if total_weights <= 0:
            raise ValueError("Sum of weights must be positive")

        self.weights = [w / total_weights for w in weights]

    def forward(self, logits, targets):
        total, _ = self.forward_with_components(logits, targets)
        return total

    def forward_with_components(self, logits, targets):
        raw = self.get_raw_losses(logits, targets)
        weighted = [r * w for r, w in zip(raw.values(), self.weights)]
        return sum(weighted), raw

    def get_raw_losses(self, logits, targets):
        return {
            name: lf(logits, targets)
            for name, lf in zip(self.names, self.loss_functions)
        }

    @property
    def names(self):
        return self.loss_functions_names

    @property
    def get_weights(self):
        return self.weights

    @property
    def count(self):
        return len(self.loss_functions)
