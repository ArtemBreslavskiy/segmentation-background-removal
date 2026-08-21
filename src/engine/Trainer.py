import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchmetrics
from torch.optim.lr_scheduler import LRScheduler
from datetime import datetime
from pathlib import Path
from src.engine.BaseModule import BaseModule
from src.losses.ComboLoss import ComboLoss
from src.utils.factories.loss_fn import create_loss
from src.utils.factories.metrics import create_metrics
from src.utils.factories.model import create_model
from src.utils.factories.optimizer import create_optimizer
from src.utils.factories.scheduler import create_scheduler
from src.configs.schemas.model.model import ModelConfig
from src.configs.schemas.learning.learning import LearningConfig
from src.configs.schemas.evaluating.evaluating import EvaluatingConfig


class Trainer(BaseModule):
    CORRECT_SAVE_MODE = ["min", "max"]

    def __init__(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        learning_config: LearningConfig,
        evaluating_config: EvaluatingConfig,
        loss_function: nn.Module | callable,
        optimizer: optim.Optimizer,
        log_dir: str | Path | None = None,
        metrics: dict[str, torchmetrics.Metric] | None = None,
        scheduler: LRScheduler | None = None,
        device: torch.device | str | None = None,
        model_name: str | None = None,
        logger: logging.Logger | None = None,
    ):
        super().__init__(
            model=model,
            model_config=model_config,
            learning_config=learning_config,
            loss_function=loss_function,
            optimizer=optimizer,
            log_dir=log_dir,
            metrics=metrics,
            device=device,
            model_name=model_name,
            logger=logger,
        )
        self.evaluating_config = self._validate_config(evaluating_config)
        if scheduler is not None:
            self.scheduler = self._validate_scheduler(scheduler)
        else:
            self.scheduler = None
        self.optimizer = self._validate_optimizer(optimizer)

        self.save_criterion = None
        self.patience_counter = 0
        self.best_value = None
        self.interrupted_in = None
        self.metrics_history = {
            "train": {name: [] for name in self.metrics.keys()},
            "val": {name: [] for name in self.metrics.keys()},
        }

    @staticmethod
    def _validate_scheduler(scheduler: LRScheduler) -> LRScheduler:
        if scheduler is None:
            raise ValueError("scheduler cannot be None")
        return scheduler

    def _validate_fit_epochs(self, epochs: int) -> int:
        remaining_epochs = epochs - self.current_epoch
        if remaining_epochs < 1:
            raise ValueError(
                f"No epochs left to train: total_epochs={epochs + self.current_epoch}, "
                f"current_epoch={self.current_epoch}"
            )
        return remaining_epochs

    @staticmethod
    def _validate_fit_mode(mode: str) -> str:
        mode = mode.lower()
        if mode not in Trainer.CORRECT_SAVE_MODE:
            raise ValueError(f"Unknown mode: {mode}")
        return mode

    @staticmethod
    def _validate_fit_log_interval(log_interval: int) -> int:
        if log_interval != -1 and log_interval < 1:
            raise ValueError("log_interval cannot be less than 1")
        return log_interval

    @staticmethod
    def _validate_val_save_criterion(save_criterion: str, val_dataloader: data.DataLoader | None) -> str:
        save_criterion = save_criterion.lower()
        if save_criterion.startswith("val/") and val_dataloader is None:
            raise ValueError(f"With save criterion {save_criterion}, val_dataloader cannot be None")
        return save_criterion

    @staticmethod
    def _validate_train_dataloader(dataloader: data.DataLoader) -> data.DataLoader:
        if dataloader is None:
            raise ValueError("train_dataloader cannot be none")
        return dataloader

    @staticmethod
    def _validate_val_dataloader(dataloader: data.DataLoader) -> data.DataLoader:
        if dataloader is None:
            raise ValueError("val_dataloader cannot be none")
        return dataloader

    def train_epoch(self, dataloader: data.DataLoader, resume_batches: int = 0) -> dict[str, float]:
        self._validate_train_dataloader(dataloader)
        self.interrupted_in = "train"
        metrics_values = self.run_epoch(dataloader, mode="train", resume_batches=resume_batches)
        for name, value in metrics_values.items():
            self.metrics_history["train"].setdefault(name, []).append(value)
        self.interrupted_in = None
        return metrics_values

    def validate_epoch(self, dataloader: data.DataLoader) -> dict[str, float]:
        self._validate_val_dataloader(dataloader)
        self.interrupted_in = "val"
        metrics_values = self.run_epoch(dataloader, mode="val")
        for name, value in metrics_values.items():
            self.metrics_history["val"].setdefault(name, []).append(value)
        self.interrupted_in = None
        return metrics_values

    def fit(
        self,
        train_dataloader: data.DataLoader,
        val_dataloader: data.DataLoader | None = None,
        epochs: int = 10,
        resume_batches: int = 0,
        save_criterion: str = "train/loss",
        mode: str = "min",
        early_stopping_patience: int | None = None,
        log_interval: int = 1,
    ):
        train_dataloader = self._validate_train_dataloader(train_dataloader)
        if val_dataloader is not None:
            val_dataloader = self._validate_val_dataloader(val_dataloader)
            self.save_criterion = self._validate_val_save_criterion(save_criterion, val_dataloader)
        else:
            self.save_criterion = self._validate_val_save_criterion(save_criterion, val_dataloader)

        remaining_epochs = self._validate_fit_epochs(epochs)
        mode = self._validate_fit_mode(mode)
        log_interval = self._validate_fit_log_interval(log_interval)

        if mode == "min":
            self.best_value = float("inf") if not self.best_value else self.best_value

            def is_better(current, best):
                return current < best

        else:
            self.best_value = float("-inf") if not self.best_value else self.best_value

            def is_better(current, best):
                return current > best

        resume_incomplete_train = (self.current_batch_in_epoch > 0) and (self.interrupted_in == "train")
        resume_incomplete_val = (self.interrupted_in == "val") and (not resume_incomplete_train)
        if resume_incomplete_train or resume_incomplete_val:
            remaining_epochs += 1

        for _ in range(remaining_epochs):
            if resume_incomplete_train:
                resume_incomplete_train = False

                train_metrics = self.train_epoch(train_dataloader, resume_batches=resume_batches)
                if val_dataloader is not None:
                    val_metrics = self.validate_epoch(val_dataloader)

            elif resume_incomplete_val:
                resume_incomplete_val = False

                train_metrics = {
                    name: self.metrics_history["train"][name][-1]
                    for name in self.metrics_history["train"]
                    if self.metrics_history["train"][name]
                }
                if not train_metrics:
                    self.logger.warning("No train history found for interrupted validation, running full epoch")
                    train_metrics = self.train_epoch(train_dataloader, resume_batches=0)

                if val_dataloader is not None:
                    val_metrics = self.validate_epoch(val_dataloader)

            else:
                self.current_epoch += 1
                train_metrics = self.train_epoch(train_dataloader, resume_batches=0)
                if val_dataloader is not None:
                    val_metrics = self.validate_epoch(val_dataloader)

            if self.save_criterion.startswith("val/") and val_dataloader is not None:
                key = self.save_criterion[4:]
                if key in val_metrics:
                    current_value = val_metrics[key]
                else:
                    raise ValueError("save_criterion not found in metrics")

            elif self.save_criterion.startswith("train/"):
                key = self.save_criterion[6:]
                if key in train_metrics:
                    current_value = train_metrics[key]
                else:
                    raise ValueError("save_criterion not found in metrics")

            else:
                raise ValueError("save_criterion not found in metrics")

            if is_better(current_value, self.best_value):
                self.best_value = current_value
                self.logger.info(
                    "New best model at epoch %d: %s = %.4f. Checkpoint saved.",
                    self.current_epoch,
                    save_criterion,
                    self.best_value,
                )
                self.save_checkpoint(is_best=True)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                self.logger.debug(
                    "Epoch %d: %s = %.4f (best: %.4f, patience: %d/%s)",
                    self.current_epoch,
                    self.save_criterion,
                    current_value,
                    self.best_value,
                    self.patience_counter,
                    early_stopping_patience or "∞",
                )

            if early_stopping_patience and self.patience_counter >= early_stopping_patience:
                self.logger.warning(
                    "Early stopping triggered after %d epochs without improvement. Best %s: %.4f",
                    early_stopping_patience,
                    self.save_criterion,
                    self.best_value,
                )
                break

            if self.scheduler:
                loss_for_scheduler = val_metrics["loss"] if val_dataloader is not None else train_metrics["loss"]
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(loss_for_scheduler)
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.logger.debug(f"Scheduler stepped (epoch-based). Current LR: {current_lr:.2e}")

            if self.current_epoch % log_interval == 0 and log_interval != -1:
                self.logger.info("Checkpoint saved for epoch %d (intermediate).", self.current_epoch)
                self.interrupted_in = None
                self.save_checkpoint(is_best=False)

    def save_checkpoint(self, is_best: bool = False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if is_best:
            filename = self.log_dir / f"{self.model_name}_best.pt"
        else:
            (self.log_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            filename = self.log_dir / "checkpoints" / f"{self.model_name}_epoch_{self.current_epoch}_{timestamp}.pt"

        checkpoint = {
            "model_config": self.model_config,
            "learning_config": self.learning_config,
            "evaluating_config": self.evaluating_config,

            "model_state_dict": self.model.state_dict(),
            "model_name": self.model_name,

            "epoch": self.current_epoch,
            "current_batch_in_epoch": self.current_batch_in_epoch,
            "interrupted_in": self.interrupted_in,
            "save_criterion": self.save_criterion,
            "patience_counter": self.patience_counter,
            "best_value": self.best_value,
            "metrics_history": self.metrics_history,

            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (self.scheduler.state_dict() if self.scheduler else None),
        }
        torch.save(checkpoint, filename)
        self.logger.debug("Checkpoint saved to %s", filename)

    def load_checkpoint(
        self,
        path: str | Path,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        logger: logging.Logger | None = None,
    ) -> dict[str, any]:
        if logger:
            self.logger = logger
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.logger.info(
            "Checkpoint loaded from %s, resuming from epoch %d",
            path,
            self.current_epoch,
        )

        self.model_config = checkpoint["model_config"]
        self.learning_config = checkpoint["learning_config"]
        self.evaluating_config = checkpoint["evaluating_config"]

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model_name = checkpoint["model_name"]

        self.current_epoch = checkpoint["epoch"]
        self.current_batch_in_epoch = checkpoint.get("current_batch_in_epoch", 0)
        self.interrupted_in = checkpoint.get("interrupted_in", None)
        self.save_criterion = checkpoint["save_criterion"]
        self.patience_counter = checkpoint.get("patience_counter", 0)
        self.best_value = checkpoint["best_value"]
        self.metrics_history = checkpoint["metrics_history"]

        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if load_scheduler and checkpoint["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint

    @classmethod
    def load_trainer(
        cls,
        path: str | Path,
        log_dir: str | Path | None = None,
        device: torch.device | str | None = None,
        logger: logging.Logger | None = None,
    ):
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        model_config = checkpoint["model_config"]
        model_config = BaseModule._validate_config(model_config)
        learning_config = checkpoint["learning_config"]
        learning_config = BaseModule._validate_config(learning_config)
        evaluating_config = checkpoint["evaluating_config"]
        evaluating_config = BaseModule._validate_config(evaluating_config)

        model = create_model(model_config)
        loss_function = create_loss(learning_config)
        metrics = create_metrics(evaluating_config)
        optimizer = create_optimizer(learning_config, model)
        scheduler = create_scheduler(learning_config, optimizer)

        trainer = cls(
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            model_config=model_config,
            learning_config=learning_config,
            evaluating_config=evaluating_config,
            log_dir=log_dir,
            metrics=metrics,
            scheduler=scheduler,
            device=device,
            model_name=checkpoint["model_name"],
            logger=logger,
        )
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if trainer.scheduler and checkpoint["scheduler_state_dict"]:
            trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        trainer.metrics = {name: metric.to(trainer.device) for name, metric in trainer.metrics.items()}
        trainer.metrics_history = checkpoint["metrics_history"]
        trainer.current_epoch = checkpoint["epoch"]
        trainer.best_value = checkpoint["best_value"]
        trainer.save_criterion = checkpoint["save_criterion"]
        trainer.has_components = isinstance(trainer.loss_function, ComboLoss)

        return trainer
