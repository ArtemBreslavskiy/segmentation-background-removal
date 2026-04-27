import json
import logging
from typing import Dict, Optional

import torch
import yaml

from ProjectPaths import ProjectPaths
from src.engine.Tester import Tester
from src.logs.logger_setup import configure_loggers
from src.utils.factories.dataloader_factory import (
    create_test_dataloader_with_weighted_dynamic_bucket_batch_sampler,
)
from src.utils.factories.metrics_factory import create_metrics
from src.utils.weighted_dynamic_bucket_batch_sampler_utils import get_padding_fn


def evaluate(config: Dict, logger: Optional[logging.Logger] = None):
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("EVALUATION STARTED")
    logger.info("=" * 60)

    path = ProjectPaths()
    with open(path.TEST) as f:
        test_manifest = json.load(f)
    logger.debug("Manifest loaded successfully")

    model_name = config["model"]["model_name"]
    saved_path = path.SAVED_CHECKPOINTS / f"{model_name}_best.pt"
    log_dir = path.SAVED_BEST_MODEL_TESTS
    device = "cuda" if config["learning"]["use_cuda"] and torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    if device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info(
            "GPU Memory: %.2f GB",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
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
        collate_fn = get_padding_fn(config)

        val_loader = create_test_dataloader_with_weighted_dynamic_bucket_batch_sampler(
            config=config, manifest=test_manifest, collate_fn=collate_fn, shuffle=False
        )
        logger.info("Test dataloader created")

        metrics = create_metrics(config)
        logger.info("Metrics initialized: %s", list(metrics.keys()))

        data = tester.evaluate(dataloader=val_loader, metrics=metrics)

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
