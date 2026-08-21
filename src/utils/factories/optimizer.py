import torch.nn as nn
from src.configs.schemas.learning.optimizer import BaseOptimizerConfig


def create_optimizer(config: BaseOptimizerConfig, model: nn.Module):
    if config.type == "torch_adam_w":
        from torch.optim import AdamW
        optimizer_params = config.model_dump(exclude={"type"})
        return AdamW(model.parameters(), **optimizer_params)

    else:
        raise ValueError(f"Unknown optimizer type: {config.type}")
