import json
import logging
import torch
from paths.project_paths import ProjectPaths
from src.engine.Tester import Tester
from src.utils.logger_setup import get_logger, get_null_logger
from src.utils.factories.dataset import create_test_dataset
from src.utils.factories.dataloader import create_test_dataloader
from src.utils.factories.batch_sampler import create_batch_sampler
from src.utils.factories.metrics import create_metrics
from src.utils.sampler_utils import get_padding_fn
from src.configs.schemas.model.model import ModelConfig
from src.configs.schemas.evaluating.evaluating import EvaluatingConfig
from src.configs.schemas.dataset.dataset import BinarySegmentationDatasetConfig
from src.configs.schemas.dataloader.dataloader import DataloaderConfig
from src.configs.loader import load_model_config, load_evaluating_config, load_dataset_config, load_dataloader_config


def evaluate(
    model_config: ModelConfig,
    evaluating_config: EvaluatingConfig,
    dataset_config: BinarySegmentationDatasetConfig,
    dataloader_config: DataloaderConfig,
    logger: logging.Logger | None = None
):
    try:
        if logger is None:
            logger = get_null_logger()

        logger.info("=" * 60)
        logger.info("EVALUATION STARTED")
        logger.info("=" * 60)

        path = ProjectPaths()
        with open(path.TEST) as f:
            test_manifest = json.load(f)
        logger.debug("Manifest loaded successfully")

        model_name = model_config.model_name
        saved_path = path.SAVED_CHECKPOINTS / f"{model_name}_best.pt"
        log_dir = path.SAVED_BEST_MODEL_TESTS
        device = "cuda" if evaluating_config.use_cuda and torch.cuda.is_available() else "cpu"
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
            raise FileNotFoundError(f"Checkpoint not found at: {saved_path}")

        logger.info("Starting evaluation on test dataset...")
        collate_fn = get_padding_fn(dataloader_config.pad_collate)
        test_dataset = create_test_dataset(manifest=test_manifest, config=dataset_config)
        batch_sampler = create_batch_sampler(config=dataloader_config.samplers.test, dataset=test_dataset)
        test_loader = create_test_dataloader(
            config=dataloader_config,
            dataset=test_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn
        )
        logger.info("Test dataloader created")

        metrics = create_metrics(config=evaluating_config)
        logger.info("Metrics initialized: %s", list(metrics.keys()))

        data = tester.evaluate(dataloader=test_loader, metrics=metrics)

        logger.info("=" * 60)
        logger.info("TEST METRICS RESULTS")
        logger.info("=" * 60)
        for key, value in data.items():
            logger.info("%s: %s", key, value)
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

    except Exception as ex:
        logger.exception(f"Error during evaluation: {ex}")
        raise


if __name__ == "__main__":
    path = ProjectPaths()
    model_conf = load_model_config(path.MODEL_CONFIG)
    evaluating_conf = load_evaluating_config(path.EVALUATING_CONFIG)
    dataset_conf = load_dataset_config(path.DATASET_CONFIG)
    dataloader_conf = load_dataloader_config(path.DATALOADER_CONFIG)
    evaluate_logger = get_logger("evaluating")
    evaluate(
        model_config=model_conf,
        evaluating_config=evaluating_conf,
        dataset_config=dataset_conf,
        dataloader_config=dataloader_conf,
        logger=evaluate_logger,
    )
