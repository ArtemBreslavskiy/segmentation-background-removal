import torch.nn as nn
from typing import Dict
from src.utils.factories.factory_utils import get_class, convert_value


def create_optimizer(config: Dict, model: nn.Module):
    OptimizerClass = get_class(config["learning"]["optimizer"]["class"])
    return OptimizerClass(
        model.parameters(), **convert_value(config["learning"]["optimizer"]["params"])
    )