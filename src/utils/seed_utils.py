import random
import numpy as np
import torch
from src.configs.schemas.dataloader.dataloader import DataloaderConfig


def set_random_seed(config: DataloaderConfig):
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
