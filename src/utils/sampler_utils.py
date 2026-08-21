from functools import partial
from src.data.pad_collate import pad_collate
from src.configs.schemas.dataloader.pad_collate import PadCollateConfig


def get_area_and_aspect_ratio(h: int, w: int) -> tuple[int, float]:
    return h * w, h / w


def get_sample_weight(dataset_weights: dict[str, int], source: str) -> float:
    return dataset_weights.get(source, 0)


def get_padding_fn(config: PadCollateConfig) -> partial | None:
    if config.enabled:
        collate_fn = partial(
            pad_collate,
            alignment=config.alignment,
            pad_value=config.pad_value,
            mode=config.mode,
        )
    else:
        collate_fn = None
    return collate_fn
