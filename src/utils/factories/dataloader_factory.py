import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.utils.data as data

from ProjectPaths import ProjectPaths
from src.utils.factories.batch_sampler_factory import create_batch_sampler
from src.utils.factories.dataset_factory import (
    create_test_dataset,
    create_train_dataset,
    create_val_dataset,
)
from src.utils.weighted_dynamic_bucket_batch_sampler_utils import (
    get_area_and_aspect_ratio,
    get_sample_weight,
)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_dataloader(
    config: Dict,
    mode: str,
    json_path: Optional[Union[ProjectPaths, str, Path]] = None,
    manifest: Optional[List[Dict]] = None,
    batch_sampler: Optional[data.BatchSampler] = None,
    collate_fn: Optional[Callable] = None,
    resize_mode: str = "resize",
) -> data.DataLoader:
    mode = mode.lower()
    correct_modes = ["train", "test", "val"]
    if mode not in correct_modes:
        raise ValueError(f"Unknown mode: {mode}. Available mods: {correct_modes}")

    if isinstance(json_path, ProjectPaths):
        if mode == "train":
            json_path = json_path.TRAIN
        elif mode == "test":
            json_path = json_path.TEST
        elif mode == "val":
            json_path = json_path.VAL

    if not manifest:
        if json_path:
            with open(json_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        else:
            raise ValueError("Either json_path or manifest must be provided")

    dataset = None
    dataloader_config = config["dataloader"]
    batch_size = dataloader_config["batch_sizes"][mode]
    num_workers = dataloader_config.get("num_workers", 0)
    shuffle = dataloader_config["shuffle"][mode]

    if batch_sampler is not None:
        if batch_size is not None or shuffle is not None:
            raise ValueError(
                "When batch_sampler is provided, batch_size and shuffle must be None/null in config."
            )
    elif batch_size is None:
        raise ValueError(
            "batch_size must be set in config when batch_sampler is not used."
        )
    elif shuffle is None:
        raise ValueError(
            "shuffle must be set in config when batch_sampler is not used."
        )

    if mode == "train":
        dataset = create_train_dataset(config, manifest=manifest, resize_mode=resize_mode)
    elif mode == "val":
        dataset = create_val_dataset(config, manifest=manifest, resize_mode=resize_mode)
    elif mode == "test":
        dataset = create_test_dataset(config, manifest=manifest, resize_mode=resize_mode)

    generator = torch.Generator()
    generator.manual_seed(config["dataloader"]["seed"])

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
                "persistent_workers": config.get("dataloader", {}).get(
                    "persistent_workers", False
                ),
                "prefetch_factor": config.get("dataloader", {}).get(
                    "prefetch_factor", 2
                ),
            }
        )

    return data.DataLoader(dataset, **loaded_kwargs)


def create_train_dataloader(
    config: Dict,
    json_path: Optional[Union[ProjectPaths, str, Path]] = None,
    manifest: Optional[List[Dict]] = None,
    batch_sampler: Optional[data.BatchSampler] = None,
    collate_fn: Optional[Callable] = None,
    resize_mode: str = "resize",
) -> data.DataLoader:
    return create_dataloader(
        config=config,
        json_path=json_path,
        manifest=manifest,
        mode="train",
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        resize_mode=resize_mode,
    )


def create_test_dataloader(
    config: Dict,
    json_path: Optional[Union[ProjectPaths, str, Path]] = None,
    manifest: Optional[List[Dict]] = None,
    batch_sampler: Optional[data.BatchSampler] = None,
    collate_fn: Optional[Callable] = None,
    resize_mode: str = "resize",
) -> data.DataLoader:
    return create_dataloader(
        config=config,
        json_path=json_path,
        manifest=manifest,
        mode="test",
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        resize_mode=resize_mode,
    )


def create_val_dataloader(
    config: Dict,
    json_path: Optional[Union[ProjectPaths, str, Path]] = None,
    manifest: Optional[List[Dict]] = None,
    batch_sampler: Optional[data.BatchSampler] = None,
    collate_fn: Optional[Callable] = None,
    resize_mode: str = "resize",
) -> data.DataLoader:
    return create_dataloader(
        config=config,
        json_path=json_path,
        manifest=manifest,
        mode="val",
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        resize_mode=resize_mode,
    )


def create_dataloader_with_weighted_dynamic_bucket_batch_sampler(
    config: Dict,
    mode: str,
    manifest: List[Dict],
    collate_fn: Callable,
    shuffle: bool = True,
    resize_mode: str = "resize",
) -> data.DataLoader:
    mode = mode.lower()
    correct_modes = ["train", "test", "val"]
    if mode not in correct_modes:
        raise ValueError(f"Unknown mode: {mode}. Available mods: {correct_modes}")

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
        "manifest": manifest,
        "batch_sampler": batch_sampler,
        "resize_mode": resize_mode,
    }
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn
    return create_dataloader(**loader_kwargs)


def create_train_dataloader_with_weighted_dynamic_bucket_batch_sampler(
    config: Dict,
    manifest: List[Dict],
    collate_fn: Callable,
    shuffle: bool = True,
    resize_mode: str = "resize",
) -> data.DataLoader:
    return create_dataloader_with_weighted_dynamic_bucket_batch_sampler(
        config=config,
        mode="train",
        manifest=manifest,
        collate_fn=collate_fn,
        shuffle=shuffle,
        resize_mode=resize_mode,
    )


def create_test_dataloader_with_weighted_dynamic_bucket_batch_sampler(
    config: Dict,
    manifest: List[Dict],
    collate_fn: Callable,
    shuffle: bool = False,
    resize_mode: str = "resize",
) -> data.DataLoader:
    return create_dataloader_with_weighted_dynamic_bucket_batch_sampler(
        config=config,
        mode="test",
        manifest=manifest,
        collate_fn=collate_fn,
        shuffle=shuffle,
        resize_mode=resize_mode,
    )


def create_val_dataloader_with_weighted_dynamic_bucket_batch_sampler(
    config: Dict,
    manifest: List[Dict],
    collate_fn: Callable,
    shuffle: bool = False,
    resize_mode: str = "resize",
) -> data.DataLoader:
    return create_dataloader_with_weighted_dynamic_bucket_batch_sampler(
        config=config,
        mode="val",
        manifest=manifest,
        collate_fn=collate_fn,
        shuffle=shuffle,
        resize_mode=resize_mode,
    )
