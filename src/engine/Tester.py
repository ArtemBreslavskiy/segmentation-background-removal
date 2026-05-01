import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.utils.data as data
import torchmetrics

from src.engine.BaseModule import BaseModule


class Tester(BaseModule):
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        loss_function: Union[nn.Module, Callable],
        metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        log_dir: Optional[Union[str, Path]] = None,
        device: Optional[Union[torch.device, str]] = None,
        model_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(
            model=model,
            config=config,
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
        metrics: Optional[Dict[str, torchmetrics.Metric]] = None
    ) -> Dict[str, float]:

        if dataloader is None:
            error_msg = "dataloader cannot be none"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)
        if not isinstance(dataloader, data.DataLoader):
            error_msg = "Unsupported dataloader type"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)
        if len(dataloader) == 0:
            error_msg = "dataloader cannot be empty"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)

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

    def _save_metrics(self, metrics: Dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_dir / f"{self.model_name}_test_{self.current_epoch+1}_{timestamp}.pt"
        torch.save(metrics, filename)
        self.logger.debug("Test metrics saved to %s", filename)

    @classmethod
    def load_tester(
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

        from src.utils.factories.loss_fn_factory import create_loss
        from src.utils.factories.metrics_factory import create_metrics
        from src.utils.factories.model_factory import create_model

        model = create_model(config)
        loss_function = create_loss(config)
        metrics = create_metrics(config)

        tester = cls(
            model=model,
            loss_function=loss_function,
            config=config,
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
