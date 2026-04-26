import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchmetrics
from torch.optim.lr_scheduler import _LRScheduler

from src.engine.BaseModule import BaseModule
from src.losses.ComboLoss import ComboLoss


class Trainer(BaseModule):
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        loss_function: Union[nn.Module, Callable],
        optimizer: Optional[optim.Optimizer] = None,
        log_dir: Optional[Union[str, Path]] = None,
        metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        scheduler: Optional[_LRScheduler] = None,
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
        self.current_epoch = 0
        self.save_criterion = None
        self.best_value = None
        self.scheduler = scheduler
        self.metrics_history = {
            "train": {name: [] for name in self.metrics.keys()},
            "val": {name: [] for name in self.metrics.keys()},
        }

    def train_epoch(self, dataloader: data.DataLoader) -> Dict[str, float]:
        metrics_values = self.run_epoch(dataloader, mode="train")
        for name, value in metrics_values.items():
            self.metrics_history["train"].setdefault(name, []).append(value)
        return metrics_values

    def validate_epoch(self, dataloader: data.DataLoader) -> Dict[str, float]:
        metrics_values = self.run_epoch(dataloader, mode="val")
        for name, value in metrics_values.items():
            self.metrics_history["val"].setdefault(name, []).append(value)
        return metrics_values

    def fit(
        self,
        train_dataloader: data.DataLoader,
        val_dataloader: Optional[data.DataLoader] = None,
        epochs: int = 10,
        save_criterion: str = "val/loss",
        mode: str = "min",
        early_stopping_patience: Optional[int] = None,
        log_interval: int = 1,
    ):
        if len(train_dataloader) == 0:
            error_msg = "Train dataloader is empty"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)

        if val_dataloader and len(val_dataloader) == 0:
            error_msg = "Val dataloader is empty"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)

        if mode not in ["min", "max"]:
            error_msg = f"Unknown mode: {mode}"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)

        patience_counter = 0
        self.save_criterion = save_criterion.lower()

        if self.save_criterion.startswith("val/") and val_dataloader is None:
            error_msg = (
                f"With save criterion {self.save_criterion}, "
                f"val_dataloader cannot be None"
            )
            self.logger.exception(error_msg)
            raise ValueError(error_msg)

        if mode == "min":
            self.best_value = float("inf") if not self.best_value else self.best_value

            def is_better(current, best):
                return current < best

        else:
            self.best_value = float("-inf") if not self.best_value else self.best_value

            def is_better(current, best):
                return current > best

        for _ in range(epochs):
            self.current_epoch += 1
            train_metrics = self.train_epoch(train_dataloader)

            if val_dataloader is not None:
                val_metrics = self.validate_epoch(val_dataloader)

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
                    "Early stopping triggered after %d epochs without improvement. "
                    "Best %s: %.4f",
                    early_stopping_patience,
                    self.save_criterion,
                    self.best_value,
                )
                break

            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(
                        val_metrics["loss"]
                        if val_dataloader is not None
                        else train_metrics["loss"]
                    )
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.logger.debug(
                    f"Scheduler stepped (epoch-based). Current LR: {current_lr:.2e}"
                )

            if self.current_epoch % log_interval == 0:
                self.logger.info(
                    "Checkpoint saved for epoch %d (intermediate).", self.current_epoch
                )
                self.save_checkpoint(is_best=False)

    def save_checkpoint(self, is_best: bool = False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if is_best:
            filename = self.log_dir / f"{self.model_name}_best.pt"
        else:
            (self.log_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            filename = (
                self.log_dir
                / "checkpoints"
                / f"{self.model_name}_epoch_{self.current_epoch}_{timestamp}.pt"
            )

        checkpoint = {
            "epoch": self.current_epoch,
            "model_name": self.model_name,
            "save_criterion": self.save_criterion,
            "best_value": self.best_value,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "metrics_history": self.metrics_history,
            "config": self.config,
            "logger": self.logger,
        }
        torch.save(checkpoint, filename)
        self.logger.debug("Checkpoint saved to %s", filename)

    def load_checkpoint(
        self,
        path: Union[str, Path],
        load_optimizer: bool = True,
        load_scheduler: bool = True,
    ):
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
        self.save_criterion = checkpoint["save_criterion"]
        self.best_value = checkpoint["best_value"]
        self.metrics_history = checkpoint["metrics_history"]
        self.logger = checkpoint["logger"]

        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if load_scheduler and checkpoint["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    @classmethod
    def load_trainer(
        cls,
        path: Union[str, Path],
        log_dir: Optional[Union[str, Path]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ):
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        config = checkpoint["config"]
        if config is None:
            raise ValueError(
                "Checkpoint does not contain config. Cannot restore components."
            )

        from src.utils.factories.loss_fn_factory import create_loss
        from src.utils.factories.metrics_factory import create_metrics
        from src.utils.factories.model_factory import create_model
        from src.utils.factories.optimizer_factory import create_optimizer
        from src.utils.factories.scheduler_factory import create_scheduler

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
        )
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if trainer.scheduler and checkpoint["scheduler_state_dict"]:
            trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        trainer.metrics = {
            name: metric.to(trainer.device) for name, metric in trainer.metrics.items()
        }
        trainer.metrics_history = checkpoint["metrics_history"]
        trainer.current_epoch = checkpoint["epoch"]
        trainer.has_components = isinstance(trainer.loss_function, ComboLoss)

        return trainer
