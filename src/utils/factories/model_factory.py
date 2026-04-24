from typing import Dict
from src.utils.factories.factory_utils import get_class, convert_value


def create_model(config: Dict):
    ModuleClass = get_class(config["model"]["class"])
    return ModuleClass(**convert_value(config["model"]["params"]))