import torch
import yaml

from paths.ProjectPaths import ProjectPaths
from scripts.build_dataset import build_processed_dataset
from scripts.evaluate import evaluate
from scripts.train import train
from src.logs.logger_setup import configure_loggers, get_logger

if __name__ == "__main__":
    path = ProjectPaths()
    with open(path.CONFIG) as f:
        config = yaml.safe_load(f)

    configure_loggers(path.CONFIG, path.LOGS)
    data_logger = get_logger(config["logs"]["types"]["data"]["name"])
    train_logger = get_logger(config["logs"]["types"]["train"]["name"])
    evaluate_logger = get_logger(config["logs"]["types"]["evaluate"]["name"])

    build_processed_dataset(config=config, logger=data_logger)
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
        train_logger.debug("Multiprocessing start method set to 'spawn'")
    except RuntimeError as ex:
        train_logger.debug("Multiprocessing start method already set: %s", ex)
    train(config=config, logger=train_logger)
    evaluate(config=config, logger=evaluate_logger)
