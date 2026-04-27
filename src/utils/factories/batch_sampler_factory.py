from typing import Dict

from src.utils.factories.factory_utils import convert_value, get_class


def create_batch_sampler(config: Dict, **override_kwargs):
    sampler_config = config["dataloader"].get("sampler")
    if sampler_config is None:
        return None
    BatchSamplerClass = get_class(sampler_config["class"])
    return BatchSamplerClass(**convert_value(sampler_config["params"]), **override_kwargs)
