import random
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
import torch.utils.data as data

from data.datasets import get_test_dataset, get_train_dataset, get_val_dataset
from paths.ProjectPaths import ProjectPaths


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_dataloader(
    config: Dict, path: Union[ProjectPaths, Path, str], mode: str
) -> data.DataLoader:
    mode = mode.lower()
    correct_modes = ["train", "val", "test"]

    if mode not in correct_modes:
        raise ValueError(f"Unknown mode: {mode}. Available mods: {correct_modes}")

    if isinstance(path, ProjectPaths):
        if mode == "train":
            path = path.TRAIN
        elif mode == "val":
            path = path.VAL
        elif mode == "test":
            path = path.TEST

    dataset = None
    batch_size = config["dataloader"]["batch_sizes"][mode]
    num_workers = config.get("dataloader", {}).get("num_workers", 0)
    shuffle = True if mode == "train" else False

    if mode == "train":
        dataset = get_train_dataset(config, path)
    elif mode == "val":
        dataset = get_val_dataset(config, path)
    elif mode == "test":
        dataset = get_test_dataset(config, path)

    generator = torch.Generator()
    generator.manual_seed(config["dataloader"]["seed"])

    loaded_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "generator": generator if shuffle else None,
    }

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
    config: Dict, path: Union[ProjectPaths, Path, str]
) -> data.DataLoader:
    return get_dataloader(config=config, path=path, mode="train")


def get_val_dataloader(
    config: Dict, path: Union[ProjectPaths, Path, str]
) -> data.DataLoader:
    return get_dataloader(config=config, path=path, mode="val")


def get_test_dataloader(
    config: Dict, path: Union[ProjectPaths, Path, str]
) -> data.DataLoader:
    return get_dataloader(config=config, path=path, mode="test")
