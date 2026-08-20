import yaml
from pathlib import Path
from src.configs.schemas.model.model import ModelConfig
from src.configs.schemas.dataset.dataset import BaseDatasetConfig
from src.configs.schemas.dataloader.dataloader import DataloaderConfig
from src.configs.schemas.learning.learning import LearningConfig
from src.configs.schemas.runtime.runtime import RuntimeConfig
from src.configs.schemas.evaluating.evaluating import EvaluatingConfig
from src.configs.schemas.logging.logging import LoggingConfig


def load_model_config(path: str | Path) -> ModelConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ModelConfig(**raw)


def load_dataset_config(path: str | Path) -> BaseDatasetConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return BaseDatasetConfig(**raw)


def load_dataloader_config(path: str | Path) -> DataloaderConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return DataloaderConfig(**raw)


def load_learning_config(path: str | Path) -> LearningConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return LearningConfig(**raw)


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return RuntimeConfig(**raw)


def load_evaluating_config(path: str | Path) -> EvaluatingConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return EvaluatingConfig(**raw)


def load_logging_config(path: str | Path) -> LoggingConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return LoggingConfig(**raw)
