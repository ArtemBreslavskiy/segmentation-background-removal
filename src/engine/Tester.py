import logging
import torch
import torch.nn as nn
import torch.utils.data as data
import torchmetrics
from datetime import datetime
from pathlib import Path
from typing import Callable
from src.engine.BaseModule import BaseModule
from src.utils.factories.loss_fn_factory import create_loss
from src.utils.factories.metrics_factory import create_metrics
from src.utils.factories.model_factory import create_model
from src.configs.schemas.model.model import ModelConfig
from src.configs.schemas.learning.learning import LearningConfig


class Tester(BaseModule):
    def __init__(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        learning_config: LearningConfig,
        loss_function: nn.Module | Callable,
        metrics: dict[str, torchmetrics.Metric] | None = None,
        log_dir: str | Path | None = None,
        device: torch.device | str | None = None,
        model_name: str | None = None,
        logger: logging.Logger | None = None,
    ):
        super().__init__(
            model=model,
            model_config=model_config,
            learning_config=learning_config,
            loss_function=loss_function,
            log_dir=log_dir,
            metrics=metrics,
            device=device,
            model_name=model_name,
            logger=logger,
        )

    def evaluate(
        self,
        dataloader: data.DataLoader,
        metrics: dict[str, torchmetrics.Metric] | None = None,
    ) -> dict[str, float]:
        self._validate_dataloader(dataloader)

        if metrics is None:
            metrics = self.metrics
        else:
            metrics = self._validate_metrics(metrics)
            metrics = {k: v.to(self.device) for k, v in metrics.items()}

        self.logger.info(f"Starting evaluation on {len(dataloader)} batches")
        self.logger.debug(f"Using metrics: {list(metrics.keys())}")

        metrics_values = self.run_epoch(dataloader, "test", metrics)

        self.logger.info("Evaluation completed")
        self._save_metrics(metrics_values)
        self.logger.debug("Results saved")
        return metrics_values

    def _save_metrics(self, metrics: dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_dir / f"{self.model_name}_test_{self.current_epoch+1}_{timestamp}.pt"
        torch.save(metrics, filename)
        self.logger.debug("Test metrics saved to %s", filename)

    @classmethod
    def load_tester(
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

        tester = cls(
            model=model,
            loss_function=loss_function,
            model_config=model_config,
            learning_config=learning_config,
            log_dir=log_dir,
            metrics=metrics,
            device=device,
            model_name=checkpoint["model_name"],
            logger=logger,
        )
        tester.model.load_state_dict(checkpoint["model_state_dict"])
        tester.metrics = {name: metric.to(tester.device) for name, metric in tester.metrics.items()}
        tester.metrics_history = checkpoint["metrics_history"]
        tester.current_epoch = checkpoint["epoch"]

        return tester
