import json
import logging
import re
from typing import Dict, Optional

import torch
import yaml

from paths.ProjectPaths import ProjectPaths
from src.engine.Trainer import Trainer
from src.logs.logger_setup import configure_loggers, get_logger
from src.utils.apply_runtime_env import apply_runtime_env
from src.utils.random_seed_utils import set_random_seed
from src.utils.factories.dataloader_factory import (
    create_train_dataloader_with_weighted_dynamic_bucket_batch_sampler,
    create_val_dataloader_with_weighted_dynamic_bucket_batch_sampler,
)
from src.utils.factories.dataset_factory import create_train_dataset, create_val_dataset
from src.utils.factories.loss_fn_factory import create_loss
from src.utils.factories.metrics_factory import create_metrics
from src.utils.factories.model_factory import create_model
from src.utils.factories.optimizer_factory import create_optimizer
from src.utils.factories.scheduler_factory import create_scheduler
from src.utils.sleep_utils import allow_sleep, prevent_sleep
from src.utils.weighted_dynamic_bucket_batch_sampler_utils import get_padding_fn


def train(config: Dict, logger: Optional[logging.Logger] = None):
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("TRAINING STARTED")
    logger.info("=" * 60)

    apply_runtime_env(config)
    set_random_seed(config)

    path = ProjectPaths()
    with open(path.TRAIN) as f:
        train_manifest = json.load(f)
    with open(path.VAL) as f:
        val_manifest = json.load(f)
    logger.debug("Manifest loaded successfully")

    model_name = config["model"]["model_name"]
    log_dir = path.SAVED_CHECKPOINTS
    device = "cuda" if config["learning"]["use_cuda"] and torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    if device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info(
            "GPU Memory: %.2f GB",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    logger.info("Model name: %s", model_name)

    logger.info("Creating model components...")

    model = create_model(config)
    logger.info("Model created: %s", config["model"]["model_name"])

    optimizer = create_optimizer(config, model)
    logger.info("Optimizer created: %s", config["learning"]["optimizer"]["class"])

    loss_function = create_loss(config)
    logger.info("Loss function created")

    metrics = create_metrics(config)
    logger.info("Metrics initialized: %s", list(metrics.keys()))

    scheduler = create_scheduler(config, optimizer)
    logger.info("Learning rate scheduler created: %s", config["learning"]["scheduler"]["class"])

    collate_fn = get_padding_fn(config)

    train_dataset = create_train_dataset(manifest=train_manifest, config=config)
    train_loader = create_train_dataloader_with_weighted_dynamic_bucket_batch_sampler(
        config=config, dataset=train_dataset, collate_fn=collate_fn, shuffle=True
    )
    logger.info("Train dataloader created")

    val_dataset = create_val_dataset(manifest=val_manifest, config=config)
    val_loader = create_val_dataloader_with_weighted_dynamic_bucket_batch_sampler(
        config=config, dataset=val_dataset, collate_fn=collate_fn, shuffle=False
    )
    logger.info("Val dataloader created")

    logger.info("Initializing Trainer...")
    try:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            metrics=metrics,
            scheduler=scheduler,
            device=device,
            log_dir=log_dir,
            model_name=model_name,
            config=config,
            logger=logger,
        )
    except Exception as ex:
        logger.exception("Trainer initialization failed: %s", ex)
        raise

    logger.info("Trainer initialized successfully")

    pattern = re.compile(rf"{re.escape(model_name)}_epoch_(\d+)_.*\.pt$")
    files = []
    for f in path.SAVED_CHECKPOINTS.iterdir():
        match = pattern.match(f.name)
        if match:
            epoch = int(match.group(1))
            files.append((epoch, f))
    if files:
        files.sort(key=lambda x: x[0])
        saved_path = files[-1][1]
        logger.info("Found existing checkpoint: %s", saved_path)
        try:
            trainer.load_checkpoint(
                path=saved_path,
                load_optimizer=True,
                load_scheduler=True,
            )
            logger.info("Checkpoint loaded successfully")
            logger.info("Resuming from epoch: %d", trainer.current_epoch)

        except Exception as ex:
            logger.exception("Failed to load checkpoint: %s", ex)
            raise
    else:
        logger.info("No checkpoint found, starting training from scratch")

    if trainer.current_batch_in_epoch > 0:
        resume_batches = trainer.current_batch_in_epoch
    else:
        resume_batches = 0

    prevent_sleep(logger)

    logger.info("=" * 60)
    logger.info("STARTING TRAINING LOOP")
    logger.info("=" * 60)
    logger.info("Total epochs: %d", config["learning"]["epochs"])
    logger.info("Save criterion: %s", config["learning"]["save_criterion"])
    logger.info("Early stopping patience: %d", config["learning"]["early_stopping_patience"])
    logger.info("Log interval: %d", config["learning"]["log_interval"])

    try:
        logger.info("Training samples: %d", len(train_loader.dataset))
        logger.info("Validation samples: %d", len(val_loader.dataset))
        trainer.fit(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=config["learning"]["epochs"],
            resume_batches=resume_batches,
            save_criterion=config["learning"]["save_criterion"],
            mode=config["learning"]["mode"],
            early_stopping_patience=config["learning"]["early_stopping_patience"],
            log_interval=config["learning"]["log_interval"],
            tqdm_mode="no_len",
        )

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        logger.info("Best %s: %.4f", trainer.save_criterion, trainer.best_value)
        trainer.save_checkpoint(is_best=False)
        raise

    except Exception as ex:
        logger.exception("Fatal error during training: %s", ex)
        logger.info("Saving checkpoint before exit due to error")
        trainer.save_checkpoint(is_best=False)
        raise

    finally:
        allow_sleep(logger)


if __name__ == "__main__":
    path = ProjectPaths()
    with open(path.CONFIG) as f:
        config = yaml.safe_load(f)

    configure_loggers(path.CONFIG, path.LOGS)
    train_logger = get_logger(config["logs"]["types"]["train"]["name"])

    train(config=config, logger=train_logger)
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
        train_logger.debug("Multiprocessing start method set to 'spawn'")
    except RuntimeError as ex:
        train_logger.debug("Multiprocessing start method already set: %s", ex)
