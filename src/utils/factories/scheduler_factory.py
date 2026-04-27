from typing import Dict

import torch.optim as optim

from src.utils.factories.factory_utils import convert_value, get_class


def create_scheduler(config: Dict, optimizer: optim.Optimizer):
    SchedulerClass = get_class(config["learning"]["scheduler"]["class"])
    return SchedulerClass(optimizer, **convert_value(config["learning"]["scheduler"]["params"]))
