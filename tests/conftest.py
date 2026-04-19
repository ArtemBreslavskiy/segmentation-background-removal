from typing import Dict
from unittest.mock import MagicMock

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

from src.data.BinarySegmentationDataset import BinarySegmentationDataset
from src.engine.Tester import Tester
from src.engine.Trainer import Trainer


class DummyDataset(data.Dataset):
    def __init__(self, size=16, image_shape=(1, 32, 32)):
        self.size = size
        self.image_shape = image_shape

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randn(self.image_shape)
        y = torch.rand(self.image_shape).round()
        return x, y


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 1, 1)
        )

    def forward(self, x):
        return self.net(x)


@pytest.fixture
def binary_dataset_path(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    for i in range(1, 4):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(
            str(images_dir / f"img_{i}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        )
        mask = np.random.choice([0, 255], (64, 64), p=[0.7, 0.3]).astype(np.uint8)
        cv2.imwrite(str(masks_dir / f"mask_{i}.png"), mask)
    return tmp_path


@pytest.fixture
def mismatched_binary_path(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.imwrite(str(images_dir / "img1.png"), img)
    cv2.imwrite(str(images_dir / "img2.png"), img)
    cv2.imwrite(str(masks_dir / "mask1.png"), mask)
    return tmp_path


@pytest.fixture
def missing_mask_path(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(images_dir / "img1.png"), img)
    cv2.imwrite(str(images_dir / "img2.png"), img)
    mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.imwrite(str(masks_dir / "mask1.png"), mask)
    with open(str(masks_dir / "mask2.png"), "w") as f:
        f.write("not an image")
    return tmp_path


@pytest.fixture
def missing_image_path(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(images_dir / "img1.png"), img)
    with open(str(images_dir / "img2.png"), "w") as f:
        f.write("not an image")
    mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.imwrite(str(masks_dir / "mask1.png"), mask)
    cv2.imwrite(str(masks_dir / "mask2.png"), mask)
    return tmp_path


@pytest.fixture
def empty_binary_path(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    return tmp_path


@pytest.fixture
def binary_dataset(binary_dataset_path):
    return BinarySegmentationDataset(binary_dataset_path)


@pytest.fixture
def binary_dataset_with_compose(binary_dataset_path):
    transforms = A.Compose(
        [
            A.RandomRotate90(),
            A.HorizontalFlip(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.ToTensorV2(),
        ]
    )
    return BinarySegmentationDataset(binary_dataset_path, transforms=transforms)


@pytest.fixture
def binary_dataset_with_dict(binary_dataset_path):
    transforms = {
        "geometric": A.Compose([A.RandomRotate90(), A.HorizontalFlip()]),
        "photometric": A.RandomBrightnessContrast(),
        "final": A.Compose(
            [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.ToTensorV2(),
            ]
        ),
    }
    return BinarySegmentationDataset(binary_dataset_path, transforms=transforms)


@pytest.fixture
def binary_dataset_dict_no_final(binary_dataset_path):
    transforms = {
        "geometric": A.Compose([A.RandomRotate90()]),
        "photometric": A.RandomBrightnessContrast(),
    }
    return BinarySegmentationDataset(binary_dataset_path, transforms=transforms)


class MockPaths:
    def __init__(self, base):
        self.base = base
        self.RAW_DATA = base / "raw"
        self.PROCESSED_DATA = base / "processed"
        self.SAVED_CHECKPOINTS = base / "checkpoints"
        self.DUTS_TR_IMAGES = self.RAW_DATA / "DUTS-TR" / "images"
        self.DUTS_TR_MASKS = self.RAW_DATA / "DUTS-TR" / "masks"
        self.DUTS_TE_IMAGES = self.RAW_DATA / "DUTS-TE" / "images"
        self.DUTS_TE_MASKS = self.RAW_DATA / "DUTS-TE" / "masks"
        self.TRAIN_IMAGES = self.PROCESSED_DATA / "train" / "images"
        self.TRAIN_MASKS = self.PROCESSED_DATA / "train" / "masks"
        self.VAL_IMAGES = self.PROCESSED_DATA / "val" / "images"
        self.VAL_MASKS = self.PROCESSED_DATA / "val" / "masks"
        self.TEST_IMAGES = self.PROCESSED_DATA / "test" / "images"
        self.TEST_MASKS = self.PROCESSED_DATA / "test" / "masks"
        self.CONFIG = base / "config.yaml"
        self.TRAIN = self.PROCESSED_DATA / "train"
        self.VAL = self.PROCESSED_DATA / "val"
        self.TEST = self.PROCESSED_DATA / "test"


@pytest.fixture
def mock_paths(tmp_path):
    return MockPaths(tmp_path)


@pytest.fixture
def sample_files(mock_paths):
    for dir_path in [
        mock_paths.DUTS_TR_IMAGES,
        mock_paths.DUTS_TR_MASKS,
        mock_paths.DUTS_TE_IMAGES,
        mock_paths.DUTS_TE_MASKS,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

    for i in range(10):
        (mock_paths.DUTS_TR_IMAGES / f"img{i}.jpg").touch()
        (mock_paths.DUTS_TR_MASKS / f"img{i}.png").touch()

    for i in range(5):
        (mock_paths.DUTS_TE_IMAGES / f"test{i}.jpg").touch()
        (mock_paths.DUTS_TE_MASKS / f"test{i}.png").touch()

    config_content = {"dataset": {"splits": {"ratios": {"val": 0.2}, "seed": 42}}}
    with open(mock_paths.CONFIG, "w") as f:
        yaml.dump(config_content, f)

    return mock_paths


@pytest.fixture
def mock_logger():
    logger = MagicMock()
    return logger


@pytest.fixture
def config() -> Dict:
    return {"model": {"model_name": "dummy_model"}, "learning": {"threshold": 0.5}}


@pytest.fixture
def full_config(tmp_path, mock_paths):
    config = {
        "model": {
            "model_name": "test_model",
            "class": "src.models.SomeModel",
            "params": {"in_channels": 3, "out_channels": 1, "features": [64, 128, 256]},
        },
        "learning": {
            "accumulation_steps": 1,
            "use_cuda": False,
            "optimizer": {
                "class": "torch.optim.Adam",
                "params": {"lr": "0.001", "weight_decay": "0.0001"},
            },
            "scheduler": {
                "class": "torch.optim.lr_scheduler.StepLR",
                "params": {"step_size": "30", "gamma": "0.1"},
            },
            "loss": {
                "class": "torch.nn.BCEWithLogitsLoss",
                "params": {"reduction": "mean"},
            },
            "epochs": 10,
            "save_criterion": "val/loss",
            "mode": "min",
            "early_stopping_patience": 5,
            "log_interval": 1,
            "threshold": 0.5,
        },
        "dataset": {"batch_sizes": {"train": 8, "val": 8, "test": 8}},
        "dataloader": {
            "batch_sizes": {"train": 8, "val": 8, "test": 8},
            "seed": 42,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": 2,
        },
        "logs": {
            "types": {
                "train": {"name": "train_logger"},
                "errors": {"name": "errors_logger"},
                "evaluate": {"name": "evaluate_logger"},
            }
        },
        "evaluating": {
            "metrics": [
                {
                    "name": "accuracy",
                    "class": "torchmetrics.classification.BinaryAccuracy",
                    "params": {"threshold": "0.5"},
                },
                {
                    "name": "iou",
                    "class": "torchmetrics.classification.BinaryJaccardIndex",
                    "params": {"threshold": "0.5"},
                },
            ]
        },
    }
    with open(mock_paths.CONFIG, "w") as f:
        import yaml

        yaml.dump(config, f)
    return config


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def model():
    return DummyModel()


@pytest.fixture
def loss():
    return nn.BCEWithLogitsLoss()


@pytest.fixture
def metrics():
    return {
        "accuracy": torchmetrics.classification.BinaryAccuracy(),
        "iou": torchmetrics.classification.BinaryJaccardIndex(),
    }


@pytest.fixture
def optimizer(model):
    return optim.Adam(model.parameters(), lr=1e-3)


@pytest.fixture
def scheduler(optimizer):
    return optim.lr_scheduler.StepLR(optimizer, step_size=1)


@pytest.fixture
def dataset():
    return DummyDataset()


@pytest.fixture
def train_loader(dataset):
    return data.DataLoader(dataset, batch_size=4)


@pytest.fixture
def val_loader(dataset):
    return data.DataLoader(dataset, batch_size=4)


@pytest.fixture
def test_loader(dataset):
    return data.DataLoader(dataset, batch_size=4)


@pytest.fixture
def log_dir(tmp_path):
    return tmp_path / "logs"


@pytest.fixture
def trainer(model, config, loss, optimizer, scheduler, metrics, log_dir, device):
    return Trainer(
        model=model,
        config=config,
        loss_function=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        log_dir=log_dir,
        device=device,
    )


@pytest.fixture
def tester(model, config, loss, metrics, log_dir, device):
    return Tester(
        model=model,
        config=config,
        loss_function=loss,
        metrics=metrics,
        log_dir=log_dir,
        device=device,
    )


@pytest.fixture
def trained_checkpoint(tmp_path, trainer):
    checkpoint_path = tmp_path / "fake_best.pt"
    torch.save(
        {
            "epoch": 5,
            "model_name": trainer.model_name,
            "save_criterion": "val/loss",
            "best_value": float("inf"),
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": (
                trainer.scheduler.state_dict() if trainer.scheduler else None
            ),
            "metrics_history": trainer.metrics_history,
            "config": trainer.config,
        },
        checkpoint_path,
    )
    return checkpoint_path


@pytest.fixture
def mock_create_model(monkeypatch, model):
    def mock(*args, **kwargs):
        return model

    monkeypatch.setattr("src.utils.factory.create_model", mock)


@pytest.fixture
def mock_create_loss(monkeypatch, loss):
    def mock(*args, **kwargs):
        return loss

    monkeypatch.setattr("src.utils.factory.create_loss", mock)


@pytest.fixture
def mock_create_metrics(monkeypatch, metrics):
    def mock(*args, **kwargs):
        return metrics

    monkeypatch.setattr("src.utils.factory.create_metrics", mock)


@pytest.fixture
def mock_create_optimizer(monkeypatch, optimizer):
    def mock(*args, **kwargs):
        return optimizer

    monkeypatch.setattr("src.utils.factory.create_optimizer", mock)


@pytest.fixture
def mock_create_scheduler(monkeypatch, scheduler):
    def mock(*args, **kwargs):
        return scheduler

    monkeypatch.setattr("src.utils.factory.create_scheduler", mock)


@pytest.fixture
def mock_paths_with_evaluate(tmp_path):
    mock = MockPaths(tmp_path)
    mock.SAVED_BEST_MODEL_TESTS = tmp_path / "evaluation_results"
    return mock
