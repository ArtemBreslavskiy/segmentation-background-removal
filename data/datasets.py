from pathlib import Path
from typing import Dict, Union

from src.data.BinarySegmentationDataset import BinarySegmentationDataset
from src.data.transforms import get_train_transforms, get_val_test_transforms


def get_train_dataset(
    config: Dict, path: Union[Path, str]
) -> BinarySegmentationDataset:
    return BinarySegmentationDataset(
        path,
        get_train_transforms(
            config["dataset"]["image"]["height"], config["dataset"]["image"]["width"]
        ),
    )


def get_val_dataset(config: Dict, path: Union[Path, str]) -> BinarySegmentationDataset:
    return BinarySegmentationDataset(
        path,
        get_val_test_transforms(
            config["dataset"]["image"]["height"], config["dataset"]["image"]["width"]
        ),
    )


def get_test_dataset(config: Dict, path: Union[Path, str]) -> BinarySegmentationDataset:
    return BinarySegmentationDataset(
        path,
        get_val_test_transforms(
            config["dataset"]["image"]["height"], config["dataset"]["image"]["width"]
        ),
    )
