import random
import numpy as np
import torch
import torch.utils.data as data
from typing import Callable, Optional
from src.configs.schemas.dataloader.dataloader import DataloaderConfig


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_dataloader(
    config: DataloaderConfig,
    mode: str,
    dataset: data.Dataset,
    batch_sampler: data.BatchSampler | None = None,
    collate_fn: Optional[Callable] = None,
) -> data.DataLoader:
    mode = mode.lower()
    correct_modes = ["train", "test", "val"]
    if mode not in correct_modes:
        raise ValueError(f"Unknown mode: {mode}. Available mods: {correct_modes}")
    if dataset is None:
        raise ValueError("dataset cannot be none")

    dataloader_params = config.model_dump()
    batch_size = dataloader_params["batch_sizes"][mode]
    num_workers = dataloader_params.get("num_workers", 0)
    shuffle = dataloader_params["shuffle"][mode]
    generator = torch.Generator()
    generator.manual_seed(dataloader_params["seed"])

    if batch_sampler is not None:
        if batch_size is not None or shuffle is not None:
            raise ValueError("When batch_sampler is provided, batch_size and shuffle must be None/null in configs.")
    elif batch_size is None:
        raise ValueError("batch_size must be set in configs when batch_sampler is not used.")
    elif shuffle is None:
        raise ValueError("shuffle must be set in configs when batch_sampler is not used.")

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
                "pin_memory": config.pin_memory,
                "persistent_workers": config.persistent_workers,
                "prefetch_factor": config.prefetch_factor,
            }
        )

    return data.DataLoader(dataset, **loaded_kwargs)


def create_train_dataloader(
    config: DataloaderConfig,
    dataset: data.Dataset,
    batch_sampler: data.BatchSampler | None = None,
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
    config: DataloaderConfig,
    dataset: data.Dataset,
    batch_sampler: data.BatchSampler | None = None,
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
    config: DataloaderConfig,
    dataset: data.Dataset,
    batch_sampler: data.BatchSampler | None = None,
    collate_fn: Optional[Callable] = None,
) -> data.DataLoader:
    return create_dataloader(
        config=config,
        dataset=dataset,
        mode="val",
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
    )
