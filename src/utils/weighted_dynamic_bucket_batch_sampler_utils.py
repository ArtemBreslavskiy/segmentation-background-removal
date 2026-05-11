from functools import partial
from typing import Dict

from src.data.pad_collate import pad_collate


def get_area_and_aspect_ratio(h: int, w: int):
    return h * w, h / w


def get_sample_weight(dataset_weights: Dict[str, int], source: str):
    return dataset_weights.get(source, 0)


def get_padding_fn(config: Dict):
    if config["dataloader"]["pad_collate"]["enabled"]:
        collate_fn = partial(
            pad_collate,
            alignment=config["dataloader"]["pad_collate"]["alignment"],
            pad_value=config["dataloader"]["pad_collate"]["pad_value"],
            mode=config["dataloader"]["pad_collate"]["mode"],
        )
    else:
        collate_fn = None
    return collate_fn
