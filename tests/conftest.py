import logging
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchmetrics
import yaml
from albumentations.pytorch import ToTensorV2

from src.data.BinarySegmentationDataset import BinarySegmentationDataset
from src.engine.BaseModule import BaseModule
from src.engine.Tester import Tester
from src.engine.Trainer import Trainer
from src.losses.ComboLoss import ComboLoss


def batch_generator(length: int, type: str = "tuple", item: int = 2):
    images = []
    masks = []
    valids = []
    for _ in range(length):
        img = torch.randn(3, 64, 64, dtype=torch.float32)
        images.append(img)

        mask = torch.zeros(1, 64, 64, dtype=torch.float32)
        mask[20:40, 20:40] = 1.0
        masks.append(mask)

        valid_mask = torch.zeros(1, 64, 64, dtype=torch.float32)
        valid_mask[:, :32, :] = 1.0
        valids.append(valid_mask)

    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    valids = torch.stack(valids, dim=0)

    if type == "tuple":
        if item <= 1:
            return (images,)
        if item == 2:
            return (images, masks)
        if item >= 3:
            return (images, masks, valids)
    if type == "list":
        if item <= 1:
            return [
                images,
            ]
        if item == 2:
            return [images, masks]
        if item >= 3:
            return [images, masks, valids]
    if type == "dict":
        if item <= 1:
            return {"image": images}
        if item == 2:
            return {"image": images, "mask": masks}
        if item >= 3:
            return {"image": images, "mask": masks, "valid_mask": valids}


def manifest_generator(path, length: int, num_datasets: int):
    data_dir = path / "dataset"
    data_dir.mkdir()
    manifest = []
    for i in range(length):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img_path = data_dir / f"img{i}.png"
        cv2.imwrite(str(img_path), img)

        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 255
        mask_path = data_dir / f"mask{i}.png"
        cv2.imwrite(str(mask_path), mask)

        source = f"Dataset{i % num_datasets}"
        manifest.append({"image": str(img_path), "mask": str(mask_path), "source": source, "resolution": [64, 64]})
    return manifest


@pytest.fixture
def dummy_config():
    config_path = Path(__file__).parent / "dummy_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture
def dummy_manifest(tmp_path):
    return manifest_generator(tmp_path, 3, 1)


@pytest.fixture
def dummy_batch():
    img1 = torch.rand(3, 64, 64, dtype=torch.float32)
    mask1 = torch.zeros(1, 64, 64, dtype=torch.float32)
    mask1[20:40, 20:40] = 1.0

    img2 = torch.rand(3, 32, 32, dtype=torch.float32)
    mask2 = torch.zeros(1, 32, 32, dtype=torch.float32)
    mask2[20:40, 20:40] = 1.0

    return [(img1, mask1), (img2, mask2)]


@pytest.fixture
def dummy_dataset(dummy_manifest, dummy_basic_transforms):
    return BinarySegmentationDataset(manifest=dummy_manifest, transforms=dummy_basic_transforms)


@pytest.fixture
def dummy_dataloader(dummy_dataset):
    return data.DataLoader(dummy_dataset, batch_size=1)


@pytest.fixture
def dummy_empty_dataset(dummy_manifest):
    return data.TensorDataset(torch.empty(0, 3, 32, 32), torch.empty(0, 1, 32, 32))


@pytest.fixture
def dummy_empty_dataloader(dummy_empty_dataset):
    return data.DataLoader(dummy_empty_dataset, batch_size=1)


@pytest.fixture
def dummy_basic_transforms():
    return A.Compose([A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])


@pytest.fixture
def dummy_dict_transforms():
    transforms = {
        "geometric": A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.1)], additional_targets={"mask": "image"}),
        "photometric": A.Compose(
            [
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5),
            ]
        ),
        "final_image": A.Compose([A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
        "final_mask": ToTensorV2(),
    }
    return transforms


@pytest.fixture
def dummy_model():
    return nn.Sequential(
        nn.Conv2d(3, 1, kernel_size=3, padding=1),
        nn.Conv2d(3, 1, kernel_size=3, padding=1),
    )


@pytest.fixture
def dummy_loss_function():
    return nn.BCEWithLogitsLoss()


@pytest.fixture
def dummy_combo_loss_function():
    fn1 = nn.BCEWithLogitsLoss()
    fn2 = nn.MSELoss()
    return ComboLoss(loss_functions=[fn1, fn2], weights=[0.3, 0.7])


@pytest.fixture
def dummy_optimizer(dummy_model):
    return optim.Adam(params=dummy_model.parameters(), lr=1e-3)


@pytest.fixture
def dummy_scheduler(dummy_optimizer):
    return optim.lr_scheduler.ReduceLROnPlateau(dummy_optimizer)


@pytest.fixture
def dummy_metrics():
    return {
        "iou": torchmetrics.JaccardIndex(task="binary"),
        "accuracy": torchmetrics.Accuracy(task="binary"),
    }


@pytest.fixture
def dummy_logger():
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    logger.addHandler(logging.NullHandler())
    return logger


@pytest.fixture
def dummy_base_module(
    dummy_model, dummy_config, dummy_loss_function, dummy_optimizer, tmp_path, dummy_metrics, dummy_logger
) -> BaseModule:
    log_dir = tmp_path / "test_dir"
    return BaseModule(
        model=dummy_model,
        config=dummy_config,
        loss_function=dummy_loss_function,
        optimizer=dummy_optimizer,
        log_dir=log_dir,
        metrics=dummy_metrics,
        device="cpu",
        logger=dummy_logger,
    )


@pytest.fixture
def dummy_trainer(
    dummy_model,
    dummy_config,
    dummy_loss_function,
    dummy_optimizer,
    tmp_path,
    dummy_metrics,
    dummy_scheduler,
    dummy_logger,
) -> Trainer:
    log_dir = tmp_path / "test_dir"
    return Trainer(
        model=dummy_model,
        config=dummy_config,
        loss_function=dummy_loss_function,
        optimizer=dummy_optimizer,
        log_dir=log_dir,
        metrics=dummy_metrics,
        scheduler=dummy_scheduler,
        device="cpu",
        logger=dummy_logger,
    )


@pytest.fixture
def dummy_tester(dummy_model, dummy_config, dummy_loss_function, tmp_path, dummy_metrics, dummy_logger) -> Tester:
    log_dir = tmp_path / "test_dir"
    return Tester(
        model=dummy_model,
        config=dummy_config,
        loss_function=dummy_loss_function,
        log_dir=log_dir,
        metrics=dummy_metrics,
        device="cpu",
        logger=dummy_logger,
    )
