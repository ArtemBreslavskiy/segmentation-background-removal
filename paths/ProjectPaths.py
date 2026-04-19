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

    DUTS: Path = ROOT / "DUTS"
    DUTS_TR: Path = DUTS / "DUTS-TR"
    DUTS_TE: Path = DUTS / "DUTS-TE"
    DUTS_TR_IMAGES: Path = DUTS / "DUTS-TR" / "DUTS-TR-Image"
    DUTS_TR_MASKS: Path = DUTS / "DUTS-TR" / "DUTS-TR-Mask"
    DUTS_TE_IMAGES: Path = DUTS / "DUTS-TE" / "DUTS-TE-Image"
    DUTS_TE_MASKS: Path = DUTS / "DUTS-TE" / "DUTS-TE-Mask"

    PROCESSED_DATA: Path = ROOT / "data" / "processed"

    TRAIN: Path = PROCESSED_DATA / "train"
    TRAIN_IMAGES: Path = PROCESSED_DATA / "train" / "images"
    TRAIN_MASKS: Path = PROCESSED_DATA / "train" / "masks"
    VAL: Path = PROCESSED_DATA / "val"
    VAL_IMAGES: Path = PROCESSED_DATA / "val" / "images"
    VAL_MASKS: Path = PROCESSED_DATA / "val" / "masks"
    TEST: Path = PROCESSED_DATA / "test"
    TEST_IMAGES: Path = PROCESSED_DATA / "test" / "images"
    TEST_MASKS: Path = PROCESSED_DATA / "test" / "masks"

