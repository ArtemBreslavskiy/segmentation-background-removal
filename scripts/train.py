import json
import logging
import re
import torch
from paths.project_paths import ProjectPaths
from src.engine.Trainer import Trainer
from src.utils.logger_setup import get_logger, get_null_logger
from src.utils.apply_runtime_env import apply_runtime_env
from src.utils.seed_utils import set_random_seed
from src.utils.factories.dataloader import create_train_dataloader, create_val_dataloader
from src.utils.factories.batch_sampler import create_batch_sampler
from src.utils.factories.dataset import create_train_dataset, create_val_dataset
from src.utils.factories.loss_fn import create_loss
from src.utils.factories.metrics import create_metrics
from src.utils.factories.model import create_model
from src.utils.factories.optimizer import create_optimizer
from src.utils.factories.scheduler import create_scheduler
from src.utils.sampler_utils import get_padding_fn
from src.configs.schemas.model.model import ModelConfig
from src.configs.schemas.dataset.dataset import DatasetConfig
from src.configs.schemas.dataloader.dataloader import DataloaderConfig
from src.configs.schemas.learning.learning import LearningConfig
from src.configs.schemas.evaluating.evaluating import EvaluatingConfig
from src.configs.schemas.runtime.runtime import RuntimeConfig
from src.configs.loader import (
    load_model_config, load_dataset_config, load_dataloader_config,
    load_learning_config, load_evaluating_config, load_runtime_config,
)


def train(
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    dataloader_config: DataloaderConfig,
    learning_config: LearningConfig,
    evaluating_config: EvaluatingConfig,
    runtime_config: RuntimeConfig,
    logger: logging.Logger | None = None
):
    try:
        if logger is None:
            logger = get_null_logger()

        logger.info("=" * 60)
        logger.info("TRAINING STARTED")
        logger.info("=" * 60)

        apply_runtime_env(runtime_config)
        set_random_seed(dataloader_config)

        path = ProjectPaths()
        with open(path.TRAIN) as f:
            train_manifest = json.load(f)
        with open(path.VAL) as f:
            val_manifest = json.load(f)
        logger.debug("Manifest loaded successfully")

        model_name = model_config.model_name
        log_dir = path.SAVED_MODELS
        device = "cuda" if learning_config.use_cuda and torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", device)
        if device == "cuda":
            logger.info("GPU: %s", torch.cuda.get_device_name(0))
            logger.info(
                "GPU Memory: %.2f GB",
                torch.cuda.get_device_properties(0).total_memory / 1e9,
            )
        logger.info("Model name: %s", model_name)

        logger.info("Creating model components...")

        model = create_model(model_config.init)
        logger.info("Model created: %s", model_name)

        optimizer = create_optimizer(learning_config.optimizer, model)
        logger.info("Optimizer created: %s", learning_config.optimizer.type)

        loss_function = create_loss(learning_config.loss)
        logger.info("Loss function created")

        metrics = create_metrics(evaluating_config)
        logger.info("Metrics initialized: %s", list(metrics.keys()))

        scheduler = create_scheduler(learning_config.scheduler, optimizer)
        logger.info("Learning rate scheduler created: %s", learning_config.scheduler.type)

        collate_fn = get_padding_fn(dataloader_config.pad_collate)

        train_dataset = create_train_dataset(config=dataset_config, manifest=train_manifest)
        train_sampler = create_batch_sampler(config=dataloader_config.samplers.train, dataset=train_dataset)
        train_dataloader = create_train_dataloader(
            config=dataloader_config,
            dataset=train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_fn
        )
        logger.info("Train dataloader created")

        val_dataset = create_val_dataset(config=dataset_config, manifest=val_manifest)
        val_sampler = create_batch_sampler(config=dataloader_config.samplers.val, dataset=val_dataset)
        val_dataloader = create_val_dataloader(
            config=dataloader_config,
            dataset=val_dataset,
            batch_sampler=val_sampler,
            collate_fn=collate_fn
        )
        logger.info("Val dataloader created")

        logger.info("Initializing Trainer...")
        trainer = Trainer(
            model=model,
            model_config=model_config,
            learning_config=learning_config,
            evaluating_config=evaluating_config,
            optimizer=optimizer,
            loss_function=loss_function,
            metrics=metrics,
            scheduler=scheduler,
            device=device,
            log_dir=log_dir,
            model_name=model_name,
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
                logger=logger,
            )
            logger.info("Checkpoint loaded successfully")
            logger.info("Resuming from epoch: %d", trainer.current_epoch)

        except Exception as ex:
            logger.error("Failed to load checkpoint: %s", ex)
            raise
    else:
        logger.info("No checkpoint found, starting training from scratch")

    if trainer.current_batch_in_epoch > 0:
        resume_batches = trainer.current_batch_in_epoch
    else:
        resume_batches = 0

    epochs = learning_config.epochs
    save_criterion = learning_config.save_criterion
    mode = learning_config.mode
    early_stopping_patience = learning_config.early_stopping_patience
    log_interval = learning_config.log_interval

    logger.info("=" * 60)
    logger.info("STARTING TRAINING LOOP")
    logger.info("=" * 60)
    logger.info("Total epochs: %d", epochs)
    logger.info("Save criterion: %s, mode: %s", save_criterion, mode)
    logger.info("Early stopping patience: %d", early_stopping_patience)
    logger.info("Log interval: %d", log_interval)

    try:
        logger.info("Training samples: %d", len(train_dataloader.dataset))
        logger.info("Validation samples: %d", len(val_dataloader.dataset))
        trainer.fit(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epochs=epochs,
            resume_batches=resume_batches,
            save_criterion=save_criterion,
            mode=mode,
            early_stopping_patience=early_stopping_patience,
            log_interval=log_interval,
        )

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        logger.info("Best %s: %.4f", trainer.save_criterion, trainer.best_value)
        trainer.save_checkpoint(is_best=False)
        raise

    except Exception as ex:
        logger.error("Fatal error during training: %s", ex)
        logger.info("Saving checkpoint before exit due to error")
        trainer.save_checkpoint(is_best=False)
        raise


if __name__ == "__main__":
    train_logger = get_logger("train")
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
        train_logger.debug("Multiprocessing start method set to 'spawn'")
    except RuntimeError as ex:
        train_logger.debug("Multiprocessing start method already set: %s", ex)

    path = ProjectPaths()
    model_conf = load_model_config(path.MODEL_CONFIG)
    dataset_conf = load_dataset_config(path.DATASET_CONFIG)
    dataloader_conf = load_dataloader_config(path.DATALOADER_CONFIG)
    learning_conf = load_learning_config(path.LEARNING_CONFIG)
    evaluating_conf = load_evaluating_config(path.EVALUATING_CONFIG)
    runtime_conf = load_runtime_config(path.RUNTIME_CONFIG)
    train(
        model_config=model_conf,
        dataset_config=dataset_conf,
        dataloader_config=dataloader_conf,
        learning_config=learning_conf,
        evaluating_config=evaluating_conf,
        runtime_config=runtime_conf,
        logger=train_logger
    )
