import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchmetrics
from torch.optim.lr_scheduler import LRScheduler

from src.engine.BaseModule import BaseModule
from src.losses.ComboLoss import ComboLoss
from src.utils.factories.loss_fn_factory import create_loss
from src.utils.factories.metrics_factory import create_metrics
from src.utils.factories.model_factory import create_model
from src.utils.factories.optimizer_factory import create_optimizer
from src.utils.factories.scheduler_factory import create_scheduler


class Trainer(BaseModule):
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        loss_function: Union[nn.Module, Callable],
        optimizer: optim.Optimizer,
        log_dir: Optional[Union[str, Path]] = None,
        metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        scheduler: Optional[LRScheduler] = None,
        device: Optional[Union[torch.device, str]] = None,
        model_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(
            model=model,
            config=config,
            loss_function=loss_function,
            optimizer=optimizer,
            log_dir=log_dir,
            metrics=metrics,
            device=device,
            model_name=model_name,
            logger=logger,
        )
        if scheduler is not None:
            self.scheduler = self._validate_scheduler(scheduler)
        else:
            self.scheduler = None
        self.optimizer = self._validate_optimizer(optimizer)
        self.save_criterion = None
        self.best_value = None
        self.metrics_history = {
            "train": {name: [] for name in self.metrics.keys()},
            "val": {name: [] for name in self.metrics.keys()},
        }

    def _validate_scheduler(self, scheduler: LRScheduler) -> LRScheduler:
        if not isinstance(scheduler, LRScheduler):
            error_msg = "Unsupported scheduler type"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)
        return scheduler

    def _validate_fit_epochs(self, epochs: int) -> int:
        remaining_epochs = epochs - self.current_epoch
        if remaining_epochs < 1:
            error_msg = (
                f"No epochs left to train: total_epochs={epochs + self.current_epoch}, "
                f"current_epoch={self.current_epoch}"
            )
            self.logger.exception(error_msg)
            raise ValueError(error_msg)
        return remaining_epochs

    def _validate_fit_mode(self, mode: str) -> str:
        mode = mode.lower()
        if mode not in ["min", "max"]:
            error_msg = f"Unknown mode: {mode}"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)
        return mode

    def _validate_fit_log_interval(self, log_interval: int) -> int:
        if log_interval != -1 and log_interval < 1:
            error_msg = "log_interval cannot be less than 1"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)
        return log_interval

    def _validate_val_save_criterion(self, save_criterion: str) -> str:
        save_criterion = save_criterion.lower()
        if save_criterion.startswith("val/"):
            error_msg = f"With save criterion {save_criterion}, val_dataloader cannot be None"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)
        return save_criterion

    def _validate_train_dataloader(self, dataloader: data.DataLoader) -> data.DataLoader:
        if dataloader is None:
            error_msg = "train_dataloader cannot be none"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)
        if not isinstance(dataloader, data.DataLoader):
            error_msg = "Unsupported train_dataloader type"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)
        if len(dataloader.dataset) == 0:
            error_msg = "train_dataloader dataset cannot be empty"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)
        return dataloader

    def _validate_val_dataloader(self, dataloader: data.DataLoader) -> data.DataLoader:
        if dataloader is None:
            error_msg = "val_dataloader cannot be none"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)
        if not isinstance(dataloader, data.DataLoader):
            error_msg = "Unsupported val_dataloader type"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)
        if len(dataloader.dataset) == 0:
            error_msg = "val_dataloader dataset cannot be empty"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)
        return dataloader

    def train_epoch(self, dataloader: data.DataLoader, tqdm_mode: Optional[str] = "default", resume_batches: int = 0
                    ) -> Dict[str, float]:
        self._validate_train_dataloader(dataloader)
        metrics_values = self.run_epoch(dataloader, mode="train", tqdm_mode=tqdm_mode, resume_batches=resume_batches)
        for name, value in metrics_values.items():
            self.metrics_history["train"].setdefault(name, []).append(value)
        return metrics_values

    def validate_epoch(self, dataloader: data.DataLoader, tqdm_mode: Optional[str] = "default", resume_batches: int = 0
                       ) -> Dict[str, float]:
        self._validate_val_dataloader(dataloader)
        metrics_values = self.run_epoch(dataloader, mode="val", tqdm_mode=tqdm_mode, resume_batches=resume_batches)
        for name, value in metrics_values.items():
            self.metrics_history["val"].setdefault(name, []).append(value)
        return metrics_values

    def fit(
        self,
        train_dataloader: data.DataLoader,
        val_dataloader: Optional[data.DataLoader] = None,
        epochs: int = 10,
        resume_batches: int = 0,
        save_criterion: str = "train/loss",
        mode: str = "min",
        early_stopping_patience: Optional[int] = None,
        log_interval: int = 1,
        tqdm_mode: Optional[str] = "default",
    ):
        remaining_epochs = self._validate_fit_epochs(epochs)
        mode = self._validate_fit_mode(mode)
        log_interval = self._validate_fit_log_interval(log_interval)
        self._validate_train_dataloader(train_dataloader)
        if val_dataloader is not None:
            self._validate_val_dataloader(val_dataloader)
            self.save_criterion = save_criterion
        else:
            self.save_criterion = self._validate_val_save_criterion(save_criterion)

        if mode == "min":
            self.best_value = float("inf") if not self.best_value else self.best_value

            def is_better(current, best):
                return current < best

        else:
            self.best_value = float("-inf") if not self.best_value else self.best_value

            def is_better(current, best):
                return current > best

        patience_counter = 0
        for _ in range(remaining_epochs):
            self.current_epoch += 1
            train_metrics = self.train_epoch(train_dataloader, tqdm_mode=tqdm_mode, resume_batches=resume_batches)

            if val_dataloader is not None:
                val_metrics = self.validate_epoch(val_dataloader, tqdm_mode=tqdm_mode, resume_batches=resume_batches)

            if self.save_criterion.startswith("val/") and val_dataloader is not None:
                key = self.save_criterion[4:]
                if key in val_metrics:
                    current_value = val_metrics[key]
                else:
                    error_msg = "save_criterion not found in metrics"
                    self.logger.exception(error_msg)
                    raise ValueError(error_msg)

            elif self.save_criterion.startswith("train/"):
                key = self.save_criterion[6:]
                if key in train_metrics:
                    current_value = train_metrics[key]
                else:
                    error_msg = "save_criterion not found in metrics"
                    self.logger.exception(error_msg)
                    raise ValueError(error_msg)

            else:
                error_msg = "save_criterion not found in metrics"
                self.logger.exception(error_msg)
                raise ValueError(error_msg)

            if is_better(current_value, self.best_value):
                self.best_value = current_value
                self.logger.info(
                    "New best model at epoch %d: %s = %.4f. Checkpoint saved.",
                    self.current_epoch,
                    save_criterion,
                    self.best_value,
                )
                self.save_checkpoint(is_best=True)
                patience_counter = 0
            else:
                patience_counter += 1
                self.logger.debug(
                    "Epoch %d: %s = %.4f (best: %.4f, patience: %d/%s)",
                    self.current_epoch,
                    self.save_criterion,
                    current_value,
                    self.best_value,
                    patience_counter,
                    early_stopping_patience or "∞",
                )

            if early_stopping_patience and patience_counter >= early_stopping_patience:
                self.logger.warning(
                    "Early stopping triggered after %d epochs without improvement. Best %s: %.4f",
                    early_stopping_patience,
                    self.save_criterion,
                    self.best_value,
                )
                break

            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"] if val_dataloader is not None else train_metrics["loss"])
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.logger.debug(f"Scheduler stepped (epoch-based). Current LR: {current_lr:.2e}")

            if self.current_epoch % log_interval == 0 and log_interval != -1:
                self.logger.info("Checkpoint saved for epoch %d (intermediate).", self.current_epoch)
                self.save_checkpoint(is_best=False)

    def save_checkpoint(self, is_best: bool = False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if is_best:
            filename = self.log_dir / f"{self.model_name}_best.pt"
        else:
            (self.log_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            filename = self.log_dir / "checkpoints" / f"{self.model_name}_epoch_{self.current_epoch}_{timestamp}.pt"

        checkpoint = {
            "epoch": self.current_epoch,
            "current_batch_in_epoch": self.current_batch_in_epoch,
            "model_name": self.model_name,
            "save_criterion": self.save_criterion,
            "best_value": self.best_value,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (self.scheduler.state_dict() if self.scheduler else None),
            "metrics_history": self.metrics_history,
            "config": self.config,
        }
        torch.save(checkpoint, filename)
        self.logger.debug("Checkpoint saved to %s", filename)

    def load_checkpoint(
        self,
        path: Union[str, Path],
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> Dict:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.logger.info(
            "Checkpoint loaded from %s, resuming from epoch %d",
            path,
            self.current_epoch,
        )

        if self.config != checkpoint.get("config"):
            self.logger.warning(
                "Configuration mismatch detected!\n"
                "Current config differs from the one used to train the checkpoint.\n"
                "Current config hash: %s\n"
                "Checkpoint config hash: %s\n"
                "This may lead to unexpected behavior "
                "if architecture, loss, or metrics have changed.",
                hash(str(self.config)),
                hash(str(checkpoint.get("config"))),
            )
            self.config = checkpoint.get("config")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model_name = checkpoint["model_name"]
        self.current_epoch = checkpoint["epoch"]
        self.current_batch_in_epoch = checkpoint.get("current_batch_in_epoch", 0)
        self.save_criterion = checkpoint["save_criterion"]
        self.best_value = checkpoint["best_value"]
        self.metrics_history = checkpoint["metrics_history"]
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if load_scheduler and checkpoint["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint

    @classmethod
    def load_trainer(
        cls,
        path: Union[str, Path],
        log_dir: Optional[Union[str, Path]] = None,
        device: Optional[Union[torch.device, str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        config = checkpoint["config"]
        if config is None or len(config) == 0:
            raise ValueError("Checkpoint does not contain config. Cannot restore components.")

        model = create_model(config)
        loss_function = create_loss(config)
        metrics = create_metrics(config)
        optimizer = create_optimizer(config, model)
        scheduler = create_scheduler(config, optimizer)

        trainer = cls(
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            config=config,
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
