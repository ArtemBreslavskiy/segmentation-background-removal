import torch.optim as optim
from src.configs.schemas.learning.scheduler import BaseSchedulerConfig


def create_scheduler(config: BaseSchedulerConfig, optimizer: optim.Optimizer):
    if config.type == "torch_cosine_annealing_warm_restarts":
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        scheduler_params = config.model_dump(exclude={"type"})
        return CosineAnnealingWarmRestarts(optimizer, **scheduler_params)

    else:
        raise ValueError(f"Unknown scheduler type: {config.type}")
