import random
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.utils.data as data

from src.data.BinarySegmentationDataset import BinarySegmentationDataset
from src.utils.factories.batch_sampler_factory import create_batch_sampler
from src.utils.weighted_dynamic_bucket_batch_sampler_utils import (
    get_area_and_aspect_ratio,
    get_sample_weight,
)
from src.utils.factories.factory_utils import convert_value


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_dataloader(
    config: Dict,
    mode: str,
    dataset: data.Dataset,
    batch_sampler: Optional[data.BatchSampler] = None,
    collate_fn: Optional[Callable] = None,
) -> data.DataLoader:
    mode = mode.lower()
    correct_modes = ["train", "test", "val"]
    if mode not in correct_modes:
        raise ValueError(f"Unknown mode: {mode}. Available mods: {correct_modes}")

    if dataset is None:
        raise ValueError("dataset cannot be none")
    if not isinstance(dataset, data.Dataset):
        raise ValueError("Unsupported dataset format")
    if len(dataset) < 1:
        raise ValueError("dataset cannot be empty")

    dataloader_config = convert_value(config["dataloader"])
    batch_size = dataloader_config["batch_sizes"][mode]
    num_workers = dataloader_config.get("num_workers", 0)
    shuffle = dataloader_config["shuffle"][mode]
    generator = torch.Generator()
    generator.manual_seed(dataloader_config["seed"])

    if batch_sampler is not None:
        if batch_size is not None or shuffle is not None:
            raise ValueError("When batch_sampler is provided, batch_size and shuffle must be None/null in config.")
    elif batch_size is None:
        raise ValueError("batch_size must be set in config when batch_sampler is not used.")
    elif shuffle is None:
        raise ValueError("shuffle must be set in config when batch_sampler is not used.")

    loaded_kwargs = {}
    if batch_size is not None:
        loaded_kwargs["batch_size"] = batch_size
    if shuffle is not None:
        loaded_kwargs["shuffle"] = shuffle
        loaded_kwargs["generator"] = generator
    if batch_sampler is not None:
        loaded_kwargs["batch_sampler"] = batch_sampler
    if collate_fn is not None:
        loaded_kwargs["collate_fn"] = collate_fn
    if num_workers > 0:
        loaded_kwargs.update(
            {
                "num_workers": num_workers,
                "worker_init_fn": seed_worker,
                "pin_memory": config.get("dataloader", {}).get("pin_memory", True),
                "persistent_workers": config.get("dataloader", {}).get("persistent_workers", False),
                "prefetch_factor": config.get("dataloader", {}).get("prefetch_factor", 2),
            }
        )

    return data.DataLoader(dataset, **loaded_kwargs)


def create_train_dataloader(
    config: Dict,
    dataset: data.Dataset,
    batch_sampler: Optional[data.BatchSampler] = None,
    collate_fn: Optional[Callable] = None,
) -> data.DataLoader:
    return create_dataloader(
        config=config,
        dataset=dataset,
        mode="train",
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
    )


def create_test_dataloader(
    config: Dict,
    dataset: data.Dataset,
    batch_sampler: Optional[data.BatchSampler] = None,
    collate_fn: Optional[Callable] = None,
) -> data.DataLoader:
    return create_dataloader(
        config=config,
        dataset=dataset,
        mode="test",
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
    )


def create_val_dataloader(
    config: Dict,
    dataset: data.Dataset,
    batch_sampler: Optional[data.BatchSampler] = None,
    collate_fn: Optional[Callable] = None,
) -> data.DataLoader:
    return create_dataloader(
        config=config,
        dataset=dataset,
        mode="val",
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
    )


def create_dataloader_with_weighted_dynamic_bucket_batch_sampler(
    config: Dict,
    mode: str,
    dataset: BinarySegmentationDataset,
    collate_fn: Callable,
    shuffle: bool = True,
) -> data.DataLoader:
    mode = mode.lower()
    correct_modes = ["train", "test", "val"]
    if mode not in correct_modes:
        raise ValueError(f"Unknown mode: {mode}. Available mods: {correct_modes}")

    manifest = dataset.get_manifest_with_correct_resolution()
    weights = []
    dataset_areas = []
    dataset_aspect_ratios = []
    for item in manifest:
        weights.append(get_sample_weight(config["dataset"]["weights"], item["source"]))
        h, w = item["resolution"]
        area, aspect_ratio = get_area_and_aspect_ratio(h, w)
        dataset_areas.append(area)
        dataset_aspect_ratios.append(aspect_ratio)
    batch_sampler = create_batch_sampler(
        config=config,
        weights=weights,
        dataset_areas=dataset_areas,
        dataset_aspect_ratios=dataset_aspect_ratios,
        shuffle=shuffle,
    )
    loader_kwargs = {
        "mode": mode,
        "config": config,
        "dataset": dataset,
        "batch_sampler": batch_sampler,
    }
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn
    return create_dataloader(**loader_kwargs)


def create_train_dataloader_with_weighted_dynamic_bucket_batch_sampler(
    config: Dict,
    dataset: BinarySegmentationDataset,
    collate_fn: Callable,
    shuffle: bool = True,
) -> data.DataLoader:
    return create_dataloader_with_weighted_dynamic_bucket_batch_sampler(
        config=config,
        mode="train",
        dataset=dataset,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )


def create_test_dataloader_with_weighted_dynamic_bucket_batch_sampler(
    config: Dict,
    dataset: BinarySegmentationDataset,
    collate_fn: Callable,
    shuffle: bool = False,
) -> data.DataLoader:
    return create_dataloader_with_weighted_dynamic_bucket_batch_sampler(
        config=config,
        mode="test",
        dataset=dataset,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )


def create_val_dataloader_with_weighted_dynamic_bucket_batch_sampler(
    config: Dict,
    dataset: BinarySegmentationDataset,
    collate_fn: Callable,
    shuffle: bool = False,
) -> data.DataLoader:
    return create_dataloader_with_weighted_dynamic_bucket_batch_sampler(
        config=config,
        mode="val",
        dataset=dataset,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
