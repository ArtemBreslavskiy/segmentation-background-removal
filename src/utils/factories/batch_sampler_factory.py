from typing import Dict
from src.utils.factories.factory_utils import get_class, convert_value


def create_batch_sampler(
    config: Dict,
    **override_kwargs
):
    sampler_config = config["dataloader"].get("sampler")
    if sampler_config is None:
        return None
    BatchSamplerClass = get_class(sampler_config["class"])
    return BatchSamplerClass(
        **convert_value(sampler_config["params"]),
        **override_kwargs
    )