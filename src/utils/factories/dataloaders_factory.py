import random
from typing import Dict, Union, Optional, Callable
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

from src.utils.factories.datasets_factory import get_train_dataset, get_test_dataset, get_val_dataset
from paths.ProjectPaths import ProjectPaths


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_dataloader(
        config: Dict,
        json_path: Union[ProjectPaths, Path, str],
        mode: str,
        batch_sampler: Optional[data.BatchSampler] = None,
        collate_fn: Optional[Callable] = None
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

    dataset = None
    dataloader_config = config["dataloader"]
    batch_size = dataloader_config["batch_sizes"][mode]
    num_workers = dataloader_config.get("num_workers", 0)
    shuffle = dataloader_config["shuffle"][mode]

    if batch_sampler is not None:
        if batch_size is not None or shuffle is not None:
            raise ValueError("When batch_sampler is provided, batch_size and shuffle must be None/null in config.")
    elif batch_size is None:
        raise ValueError("batch_size must be set in config when batch_sampler is not used.")
    elif shuffle is None:
        raise ValueError("shuffle must be set in config when batch_sampler is not used.")

    if mode == "train":
        dataset = get_train_dataset(config, json_path)
    elif mode == "val":
        dataset = get_val_dataset(config, json_path)
    elif mode == "test":
        dataset = get_test_dataset(config, json_path)

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


def get_train_dataloader(
    config: Dict,
    json_path: Union[ProjectPaths, Path, str],
    batch_sampler: Optional[data.BatchSampler] = None,
    collate_fn: Optional[Callable] = None
) -> data.DataLoader:
    return get_dataloader(
        config=config,
        json_path=json_path,
        mode="train",
        batch_sampler=batch_sampler,
        collate_fn=collate_fn
    )


def get_test_dataloader(
    config: Dict,
    json_path: Union[ProjectPaths, Path, str],
    batch_sampler: Optional[data.BatchSampler] = None,
    collate_fn: Optional[Callable] = None
) -> data.DataLoader:
    return get_dataloader(
        config=config,
        json_path=json_path,
        mode="test",
        batch_sampler=batch_sampler,
        collate_fn=collate_fn
    )


def get_val_dataloader(
    config: Dict,
    json_path: Union[ProjectPaths, Path, str],
    batch_sampler: Optional[data.BatchSampler] = None,
    collate_fn: Optional[Callable] = None
) -> data.DataLoader:
    return get_dataloader(
        config=config,
        json_path=json_path,
        mode="val",
        batch_sampler=batch_sampler,
        collate_fn=collate_fn
    )