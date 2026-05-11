from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectPaths:
    ROOT: Path = Path(__file__).parent.parent

    CONFIG: Path = ROOT / "config.yaml"
    LOGS: Path = ROOT / "logs"
    SAVED_CHECKPOINTS: Path = ROOT / "models"
    SAVED_BEST_MODEL_TESTS: Path = SAVED_CHECKPOINTS / "best_model_tests"

    RAW_DATA: Path = ROOT / "data" / "raw"
    PROCESSED_DATA: Path = ROOT / "data" / "processed"

    TRAIN: Path = PROCESSED_DATA / "train.json"
    VAL: Path = PROCESSED_DATA / "val.json"
    TEST: Path = PROCESSED_DATA / "test.json"
