from typing import Dict

from src.utils.factories.factory_utils import convert_value, get_class


def create_model(config: Dict):
    ModuleClass = get_class(config["model"]["class"])
    return ModuleClass(**convert_value(config["model"]["params"]))
