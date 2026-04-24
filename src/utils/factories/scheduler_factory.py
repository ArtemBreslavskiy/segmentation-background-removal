import torch.optim as optim
from typing import Dict
from src.utils.factories.factory_utils import get_class, convert_value


def create_scheduler(config: Dict, optimizer: optim.Optimizer):
    SchedulerClass = get_class(config["learning"]["scheduler"]["class"])
    return SchedulerClass(
        optimizer, **convert_value(config["learning"]["scheduler"]["params"])
    )