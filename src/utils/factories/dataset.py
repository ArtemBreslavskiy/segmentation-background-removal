import json
from pathlib import Path
from paths.project_paths import ProjectPaths
from src.data.BinarySegmentationDataset import BinarySegmentationDataset
from src.data.transforms import get_train_transforms, get_val_test_transforms
from src.configs.schemas.dataset.dataset import BinarySegmentationDatasetConfig


def create_dataset(
    config: BinarySegmentationDatasetConfig,
    mode: str,
    json_path: ProjectPaths | str | Path | None = None,
    manifest: list[dict] | None = None,
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
    dataset_params = config.model_dump()

    return BinarySegmentationDataset(
        manifest=manifest,
        transforms=transforms,
        max_area=config.max_area,
        resize_mode=dataset_params["resizes"][mode].get("mode", "resize"),
        area_threshold_mix=dataset_params["resizes"][mode].get("area_threshold_mix", 0),
        min_foreground_share=dataset_params["resizes"][mode].get("min_foreground_share", 0),
    )


def create_train_dataset(
    config: BinarySegmentationDatasetConfig,
    json_path: ProjectPaths | str | Path | None = None,
    manifest: list[dict] | None = None,
) -> BinarySegmentationDataset:
    return create_dataset(config=config, mode="train", json_path=json_path, manifest=manifest)


def create_test_dataset(
    config: BinarySegmentationDatasetConfig,
    json_path: ProjectPaths | str | Path | None = None,
    manifest: list[dict] | None = None,
) -> BinarySegmentationDataset:
    return create_dataset(config=config, mode="test", json_path=json_path, manifest=manifest)


def create_val_dataset(
    config: BinarySegmentationDatasetConfig,
    json_path: ProjectPaths | str | Path | None = None,
    manifest: list[dict] | None = None,
) -> BinarySegmentationDataset:
    return create_dataset(config=config, mode="val", json_path=json_path, manifest=manifest)
