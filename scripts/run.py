import torch
import yaml

from paths.project_paths import ProjectPaths
from scripts.build_dataset import build_processed_dataset
from scripts.evaluate import evaluate
from scripts.train import train
from src.logging.logger_setup import get_logger

if __name__ == "__main__":
    path = ProjectPaths()
    with open(path.CONFIG) as f:
        config = yaml.safe_load(f)

    data_logger = get_logger("data", config["logging"])
    train_logger = get_logger("train", config["logging"])
    evaluate_logger = get_logger("evaluate", config["logging"])

    build_processed_dataset(config=config, logger=data_logger)
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
        train_logger.debug("Multiprocessing start method set to 'spawn'")
    except RuntimeError as ex:
        train_logger.debug("Multiprocessing start method already set: %s", ex)
    train(config=config, logger=train_logger)
    evaluate(config=config, logger=evaluate_logger)
