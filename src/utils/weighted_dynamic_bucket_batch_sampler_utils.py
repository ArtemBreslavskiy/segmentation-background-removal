from typing import Dict


def get_area_and_aspect_ratio(h: int, w: int):
    return h * w, h / w


def get_sample_weight(dataset_weights: Dict[str, int], source: str):
    return dataset_weights.get(source, 0)