import ctypes
import logging
import platform
from typing import Optional

import torch
import yaml

from data.dataloaders import get_train_dataloader, get_val_dataloader
from paths.ProjectPaths import ProjectPaths
from src.engine.Trainer import Trainer
from src.logs.logger_setup import configure_loggers
from src.utils.factory import (
    create_loss,
    create_metrics,
    create_model,
    create_optimizer,
    create_scheduler,
)


def train(logger: Optional[logging.Logger] = None):
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("TRAINING STARTED")
    logger.info("=" * 60)

    def prevent_sleep():
        if platform.system() == "Windows":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)
            logger.info("Sleep prevention activated (Windows)")

    def allow_sleep():
        if platform.system() == "Windows":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            logger.info("Sleep prevention deactivated (Windows)")

    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
        logger.debug("Multiprocessing start method set to 'spawn'")
    except RuntimeError as ex:
        logger.debug("Multiprocessing start method already set: %s", ex)

    path = ProjectPaths()
    logger.info("Loading configuration from: %s", path.CONFIG)

    with open(path.CONFIG) as f:
        config = yaml.safe_load(f)
    logger.debug("Configuration loaded successfully")

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
    logger.info(
        "Learning rate scheduler created: %s", config["learning"]["scheduler"]["class"]
    )

    device = (
        "cuda"
        if config["learning"]["use_cuda"] and torch.cuda.is_available()
        else "cpu"
    )
    logger.info("Using device: %s", device)
    if device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info(
            "GPU Memory: %.2f GB",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    model_name = config["model"]["model_name"]
    log_dir = path.SAVED_CHECKPOINTS

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

    saved_path = saved_path = log_dir / f"{model_name}_best.pt"
    if saved_path.exists():
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

    prevent_sleep()

    logger.info("=" * 60)
    logger.info("STARTING TRAINING LOOP")
    logger.info("=" * 60)
    logger.info("Total epochs: %d", config["learning"]["epochs"])
    logger.info("Save criterion: %s", config["learning"]["save_criterion"])
    logger.info(
        "Early stopping patience: %d", config["learning"]["early_stopping_patience"]
    )
    logger.info("Log interval: %d", config["learning"]["log_interval"])
    logger.info("Accumulation_steps: %d", config["learning"]["accumulation_steps"])

    try:
        train_loader = get_train_dataloader(config, path)
        val_loader = get_val_dataloader(config, path)

        logger.info(
            "Train dataloader created with batch size: %d",
            config["dataloader"]["batch_sizes"]["train"],
        )
        logger.info(
            "Validation dataloader created with batch size: %d",
            config["dataloader"]["batch_sizes"]["val"],
        )

        logger.info("Training samples: %d", len(train_loader.dataset))
        logger.info("Validation samples: %d", len(val_loader.dataset))

        trainer.fit(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=config["learning"]["epochs"],
            save_criterion=config["learning"]["save_criterion"],
            mode=config["learning"]["mode"],
            early_stopping_patience=config["learning"]["early_stopping_patience"],
            log_interval=config["learning"]["log_interval"],
        )

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        logger.info("Best %s: %.4f", trainer.save_criterion, trainer.best_value)
        trainer.save_checkpoint(is_best=False)
        raise

    except Exception as ex:
        logger.exception("Fatal error during training: %s", ex)
        raise

    finally:
        allow_sleep()


if __name__ == "__main__":
    from src.logs.logger_setup import get_logger

    path = ProjectPaths()
    with open(path.CONFIG) as f:
        config = yaml.safe_load(f)

    configure_loggers(path.CONFIG, path.LOGS)
    train(logger=get_logger(config["logs"]["types"]["train"]["name"]))
