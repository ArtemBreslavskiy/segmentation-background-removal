from typing import Dict

import torch.nn as nn

from src.utils.factories.factory_utils import convert_value, get_class


def create_optimizer(config: Dict, model: nn.Module):
    OptimizerClass = get_class(config["learning"]["optimizer"]["class"])
    return OptimizerClass(
        model.parameters(), **convert_value(config["learning"]["optimizer"]["params"])
    )
