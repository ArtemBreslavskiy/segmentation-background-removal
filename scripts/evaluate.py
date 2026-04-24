import logging
from typing import Optional

import torch
import yaml

from src.utils.factories.dataloaders_factory import get_test_dataloader
from paths.ProjectPaths import ProjectPaths
from src.engine.Tester import Tester
from src.logs.logger_setup import configure_loggers
from src.utils.factories.metrics_factory import create_metrics


def evaluate(logger: Optional[logging.Logger] = None):
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("EVALUATION STARTED")
    logger.info("=" * 60)

    path = ProjectPaths()
    logger.info("Loading configuration from: %s", path.CONFIG)

    with open(path.CONFIG) as f:
        config = yaml.safe_load(f)
    logger.debug("Configuration loaded successfully")

    model_name = config["model"]["model_name"]
    saved_path = path.SAVED_CHECKPOINTS / f"{model_name}_best.pt"
    log_dir = path.SAVED_BEST_MODEL_TESTS
    device = (
        "cuda"
        if config["learning"]["use_cuda"] and torch.cuda.is_available()
        else "cpu"
    )

    logger.info("Using device: %s", device)
    logger.info("Model name: %s", model_name)
    logger.info("Checkpoint path: %s", saved_path)

    if saved_path.exists():
        try:
            tester = Tester.load_tester(path=saved_path, log_dir=log_dir, device=device)
        except Exception as ex:
            logger.exception("Tester loading failed: %s", ex)
            raise

    else:
        error_msg = f"Checkpoint not found at: {saved_path}"
        logger.exception(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info("Starting evaluation on test dataset...")

    try:
        dataloader = get_test_dataloader(config, path)
        logger.info(
            "Test dataloader created with batch size: %d",
            config["dataloader"]["batch_sizes"]["test"],
        )

        metrics = create_metrics(config)
        logger.info("Metrics initialized: %s", list(metrics.keys()))

        data = tester.evaluate(dataloader=dataloader, metrics=metrics)

        logger.info("=" * 60)
        logger.info("TEST METRICS RESULTS")
        logger.info("=" * 60)
        for key, value in data.items():
            logger.info("%s: %s", key, value)
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

    except Exception:
        logger.exception("Error during evaluation")
        raise


if __name__ == "__main__":
    from src.logs.logger_setup import get_logger

    path = ProjectPaths()
    with open(path.CONFIG) as f:
        config = yaml.safe_load(f)

    configure_loggers(path.CONFIG, path.LOGS)
    evaluate(logger=get_logger(config["logs"]["types"]["evaluate"]["name"]))
