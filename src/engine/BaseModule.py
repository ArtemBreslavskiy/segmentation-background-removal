import logging
import time
from abc import ABC
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.cuda.amp import GradScaler, autocast
import torchmetrics
from tqdm import tqdm

from src.losses.ComboLoss import ComboLoss


class BaseModule(ABC):
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        loss_function: Union[nn.Module, Callable],
        optimizer: Optional[optim.Optimizer] = None,
        log_dir: Optional[Union[str, Path]] = None,
        metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        device: Optional[Union[torch.device, str]] = None,
        model_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = self._device_validate(device)
        self.model = model.to(self.device)
        self.config = config
        self.log_dir = self._ensure_log_dir(log_dir)
        self.has_components = isinstance(self.loss_function, ComboLoss)
        self.scaler = GradScaler(enabled=self.config["learning"].get("use_fp16", False))
        self.current_epoch = 0

        if metrics is None or metrics == {}:
            self.metrics = {}
        else:
            self.metrics = {
                name: metric.to(self.device) for name, metric in metrics.items()
            }
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self.config["model"].get("model_name", "model")

    def _device_validate(self, device: Union[torch.device, str, None]):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        elif isinstance(device, str):
            try:
                device = torch.device(device)
            except Exception as ex:
                error_msg = f"Invalid device parameter value: {device}. {ex}"
                self.logger.exception(error_msg)
                raise ValueError(error_msg)

        if device.type == "cuda" and not torch.cuda.is_available():
            error_msg = "GPU is not available"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)

        return device

    def _ensure_log_dir(self, log_dir):
        log_dir = Path(log_dir) if log_dir else Path.cwd() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        checkpoints_dir = log_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        return log_dir

    def _move_batch_to_device(self, batch):
        if isinstance(batch, (Tuple, List)):
            return [b.to(self.device) for b in batch]
        if isinstance(batch, Dict):
            return {key: value.to(self.device) for key, value in batch.items()}
        return batch.to(self.device)

    def _unpack_batch(self, batch: Union[Tuple, List, Dict]):
        if isinstance(batch, (Tuple, List)):
            if len(batch) >= 2:
                return batch[0], batch[1]
            else:
                error_msg = f"Batch must contain at least 2 elements, got {len(batch)}"
                self.logger.exception(error_msg)
                raise ValueError(error_msg)

        elif isinstance(batch, Dict):
            if "image" in batch and "mask" in batch:
                return batch["image"], batch["mask"]
            else:
                error_msg = (
                    f"Dictionary must contain keys 'image' and 'mask', "
                    f"got {batch.keys()}"
                )
                self.logger.exception(error_msg)
                raise ValueError(error_msg)

        elif torch.is_tensor(batch):
            error_msg = "Batch must contain both images and masks"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)

        else:
            error_msg = f"Unsupported batch type: {type(batch)}"
            self.logger.exception(error_msg)
            raise TypeError(error_msg)

    @torch.no_grad()
    def _update_metrics(
        self,
        predictions,
        targets,
        metrics: Dict[str, torchmetrics.Metric],
        threshold: Optional[float] = None,
    ):
        if predictions is None:
            error_msg = "Predictions can't be none"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)

        if targets is None:
            error_msg = "Targets can't be none"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)

        if threshold is None:
            threshold = self.config["learning"]["threshold"]

        probs = torch.sigmoid(predictions)
        preds = (probs > threshold).long()
        for name, metric in metrics.items():
            metric.update(preds, targets.long())

    def _reset_metrics(self, metrics: Dict[str, torchmetrics.Metric]):
        for metric in metrics.values():
            metric.reset()

    def _compute_metrics(self, metrics: Dict[str, torchmetrics.Metric]):
        return {name: metric.compute().item() for name, metric in metrics.items()}

    def _compute_loss(self, predictions, targets, return_components=False):
        if hasattr(self.loss_function, "forward_with_components") and return_components:
            total_loss, components = self.loss_function.forward_with_components(
                predictions, targets
            )
            return total_loss, components
        else:
            loss = self.loss_function(predictions, targets)
            if return_components:
                return loss, {"loss": loss}
            return loss

    def run_epoch(
        self,
        dataloader: data.DataLoader,
        mode: str,
        metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
    ) -> Dict[str, float]:

        if mode == "train" and self.optimizer is None:
            raise RuntimeError("Optimizer required for training mode")

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            memory_before = torch.cuda.memory_allocated() / 1024**2
            self.logger.debug(f"Memory before {mode}: {memory_before:.1f}MB")

        mode = mode.lower()
        correct_modes = ["train", "val", "test"]

        if mode not in correct_modes:
            error_msg = f"Unknown mode: {mode}. Available mods: {correct_modes}"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)

        current_metrics = (
            self.metrics if metrics is None or mode in ["train", "val"] else metrics
        )

        if dataloader is None:
            error_msg = "dataloader cannot be None"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)

        if mode == "train":
            train = True
            grad_context = nullcontext()
            self.model.train()
        else:
            train = False
            grad_context = torch.no_grad()
            self.model.eval()

        start_time = time.time()
        total_loss = 0.0
        num_batches = 0
        self._reset_metrics(current_metrics)

        accumulation_steps = self.config["learning"].get("accumulation_steps", 1)
        if accumulation_steps < 1:
            error_msg = "accumulation_steps cannot be less than 1"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)

        if self.has_components:
            total_components = {name: 0.0 for name in self.loss_function.names}
        else:
            total_components = {"loss": 0.0}

        desc = (
            "Evaluating..."
            if mode == "test"
            else f"{mode.capitalize()} Epoch {self.current_epoch}"
        )
        pbar = tqdm(dataloader, desc=desc, leave=True)

        with grad_context:
            if train:
                self.optimizer.zero_grad()
            for batch in pbar:
                try:
                    batch = self._move_batch_to_device(batch)
                    x, y = self._unpack_batch(batch)

                    with autocast(self.config["learning"].get("use_fp16", False)):
                        predictions = self.model(x)
                        loss, components = self._compute_loss(
                            predictions, y, return_components=True
                        )

                    full_loss = loss.item()

                    if train:
                        loss = loss / accumulation_steps
                        self.scaler.scale(loss).backward()
                        if (num_batches + 1) % accumulation_steps == 0:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()

                    self._update_metrics(predictions, y, current_metrics)
                    total_loss += full_loss
                    num_batches += 1
                    for name in total_components.keys():
                        total_components[name] += components[name].item()

                    pbar.set_postfix(
                        {
                            "loss": f"{full_loss:.4f}",
                            "speed": (
                                f"{num_batches / (time.time() - start_time):.1f} it/s"
                                if num_batches > 0
                                else "N/A"
                            ),
                        }
                    )
                except Exception as ex:
                    self.logger.exception(f"Error in {mode} batch {num_batches}: {ex}")
                    if mode == "train":
                        self.logger.warning(f"Skipping train batch {num_batches}")
                        try:
                            torch.cuda.empty_cache()
                        except Exception as ex:
                            self.logger.warning(ex)
                            pass
                        self.optimizer.zero_grad()
                        continue
                    else:
                        raise

            if train and num_batches % accumulation_steps != 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        if num_batches == 0:
            error_msg = "dataloader cannot be empty"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)

        if self.device.type == "cuda" and torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated() / 1024**2
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            self.logger.debug(
                f"{mode.capitalize()} memory - Final: {memory_after:.1f}MB, "
                f"Peak: {peak_memory:.1f}MB"
            )

        elapsed_time = time.time() - start_time

        self.logger.info(
            f"{mode.capitalize()} epoch completed in {elapsed_time:.2f}s, "
            f"{num_batches / elapsed_time:.2f} batches/sec"
        )

        avg_loss = total_loss / num_batches
        avg_components = {
            name: total / num_batches for name, total in total_components.items()
        }
        metrics_values = self._compute_metrics(current_metrics)

        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics_values.items()])
        self.logger.info(
            f"{mode.capitalize()} completed: loss={avg_loss:.4f}, {metrics_str}"
        )

        return {"loss": avg_loss, **metrics_values, **avg_components}
