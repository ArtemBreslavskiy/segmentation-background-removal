from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectPaths:
    ROOT: Path = Path(__file__).parent.parent

    CONFIGS: Path = ROOT / "configs"
    MODEL_CONFIG: Path = CONFIGS / "model.yaml"
    DATASET_CONFIG: Path = CONFIGS / "dataset.yaml"
    DATALOADER_CONFIG: Path = CONFIGS / "dataloader.yaml"
    LEARNING_CONFIG: Path = CONFIGS / "learning.yaml"
    RUNTIME_CONFIG: Path = CONFIGS / "runtime.yaml"
    EVALUATING_CONFIG: Path = CONFIGS / "evaluating.yaml"
    LOGGING_CONFIG: Path = CONFIGS / "logging.yaml"

    LOGS: Path = ROOT / "logging"

    SAVED_MODELS: Path = ROOT / "models"
    SAVED_CHECKPOINTS: Path = SAVED_MODELS / "checkpoints"
    SAVED_BEST_MODEL_TESTS: Path = SAVED_MODELS / "best_model_tests"

    RAW_DATA: Path = ROOT / "data" / "raw"
    PROCESSED_DATA: Path = ROOT / "data" / "processed"

    TRAIN: Path = PROCESSED_DATA / "train.json"
    VAL: Path = PROCESSED_DATA / "val.json"
    TEST: Path = PROCESSED_DATA / "test.json"
