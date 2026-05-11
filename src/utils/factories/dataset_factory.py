import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from paths.ProjectPaths import ProjectPaths
from src.data.BinarySegmentationDataset import BinarySegmentationDataset
from src.data.transforms import get_train_transforms, get_val_test_transforms
from src.utils.factories.factory_utils import convert_value


def create_dataset(
    config: Dict,
    mode: str,
    json_path: Optional[Union[ProjectPaths, str, Path]] = None,
    manifest: Optional[List[Dict]] = None,
) -> BinarySegmentationDataset:
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
    transforms = get_train_transforms() if mode == "train" else get_val_test_transforms()
    dataset_config = convert_value(config["dataset"])

    return BinarySegmentationDataset(
        manifest=manifest,
        transforms=transforms,
        max_area=dataset_config.get("max_area", 0),
        resize_mode=dataset_config["resize"][mode].get("mode", "resize"),
        area_threshold_mix=dataset_config["resize"][mode].get("area_threshold_mix", 0),
        min_foreground_share=dataset_config["resize"][mode].get("min_foreground_share", 0),
    )


def create_train_dataset(
    config: Dict,
    json_path: Optional[Union[ProjectPaths, str, Path]] = None,
    manifest: Optional[List[Dict]] = None,
) -> BinarySegmentationDataset:
    return create_dataset(config=config, mode="train", json_path=json_path, manifest=manifest)


def create_test_dataset(
    config: Dict,
    json_path: Optional[Union[ProjectPaths, str, Path]] = None,
    manifest: Optional[List[Dict]] = None,
) -> BinarySegmentationDataset:
    return create_dataset(config=config, mode="test", json_path=json_path, manifest=manifest)


def create_val_dataset(
    config: Dict,
    json_path: Optional[Union[ProjectPaths, str, Path]] = None,
    manifest: Optional[List[Dict]] = None,
) -> BinarySegmentationDataset:
    return create_dataset(config=config, mode="val", json_path=json_path, manifest=manifest)
