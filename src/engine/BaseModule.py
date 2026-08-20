import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchmetrics
from pydantic import BaseModel as PydenticBaseModel
from abc import ABC
from contextlib import nullcontext
from pathlib import Path
from typing import Callable
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from src.losses.ComboLoss import ComboLoss
from src.configs.schemas.model.model import ModelConfig
from src.configs.schemas.learning.learning import LearningConfig
from src.utils.logger_setup import get_null_logger


class BaseModule(ABC):
    CORRECT_MODES = ["train", "val", "test"]

    def __init__(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        learning_config: LearningConfig,
        loss_function: nn.Module | Callable,
        optimizer: optim.Optimizer | None = None,
        log_dir: str | Path | None = None,
        metrics: dict[str, torchmetrics.Metric] | None = None,
        device: torch.device | str | None = None,
        model_name: str | None = None,
        logger: logging.Logger | None = None,
    ):
        self.model_config = self._validate_config(model_config)
        self.learning_config = self._validate_config(learning_config)

        self.device = self._validate_device(device)
        self.loss_function = self._validate_loss_function(loss_function)
        self.optimizer = optimizer
        self.metrics = {name: metric.to(self.device) for name, metric in metrics.items()} or {}

        self.logger = logger if logger is not None else get_null_logger()
        self.log_dir = self._ensure_log_dir(log_dir)

        use_amp = self.learning_config.use_fp16 and self.device.type == "cuda"
        self.scaler = GradScaler(self.device.type, enabled=use_amp)

        self.has_components = isinstance(self.loss_function, ComboLoss)
        self.current_epoch = 0
        self.current_batch_in_epoch = 0

        model = self._validate_model(model)
        model = model.to(self.device)
        if self.learning_config.compile_model:
            model = torch.compile(
                model=model,
                dynamic=self.learning_config.compile_dynamic,
                mode=self.learning_config.compile_options
            )
        self.model = model
        self.model_name = model_name or self.model_config.model_name

    @staticmethod
    def _validate_model(model: nn.Module) -> nn.Module:
        if model is None:
            raise ValueError("model cannot be none")
        return model

    @staticmethod
    def _validate_loss_function(loss_function: nn.Module | Callable) -> nn.Module | Callable:
        if loss_function is None:
            raise ValueError("loss_function cannot be none")
        return loss_function

    @staticmethod
    def _validate_config(config: PydenticBaseModel) -> PydenticBaseModel:
        if config is None:
            raise ValueError("config cannot be none")
        return config

    @staticmethod
    def _validate_device(device: torch.device | str | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(device, str):
            try:
                return torch.device(device)
            except Exception as ex:
                raise ValueError(f"Invalid device parameter value: {device}. {ex}")

        elif isinstance(device, torch.device):
            if device.type == "cuda" and not torch.cuda.is_available():
                raise ValueError("GPU is not available")

            return device
        raise ValueError(f"Invalid device parameter value: {device}.")

    @staticmethod
    def _validate_dataloader(dataloader: data.DataLoader) -> data.DataLoader:
        if dataloader is None:
            raise ValueError("dataloader cannot be none")
        return dataloader

    @staticmethod
    def _validate_run_epoch_mode(mode: str) -> str:
        mode = mode.lower()
        if mode not in BaseModule.CORRECT_MODES:
            raise ValueError(f"Unknown mode: {mode}. Available modes: {BaseModule.CORRECT_MODES}")
        return mode

    @staticmethod
    def _validate_tensor(tensor: torch.Tensor, context: str) -> torch.Tensor:
        if tensor is None:
            raise ValueError(f"{context} cannot be none")
        elif not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Unsupported {context} format")
        return tensor

    @staticmethod
    def _validate_metrics(metrics: dict[str, torchmetrics.Metric]) -> dict[str, torchmetrics.Metric]:
        if metrics is None or metrics == {}:
            raise ValueError("metrics cannot be none or empty")
        if isinstance(metrics, torchmetrics.Metric):
            return {type(metrics).__name__: metrics}
        if isinstance(metrics, dict):
            return metrics

    @staticmethod
    def _validate_optimizer(optimizer: optim.Optimizer) -> optim.Optimizer:
        if optimizer is None:
            raise ValueError("optimizer cannot be none")
        return optimizer

    def _request_optimizer(self):
        if self.optimizer is None:
            raise RuntimeError("Optimizer required for training mode")

    def _ensure_log_dir(self, log_dir: str | Path) -> Path:
        if log_dir is None:
            log_dir = Path.cwd() / "logging"
            self.logger.warning(f"Entered log_dir is empty. Auto log_dir: {log_dir}")
        else:
            log_dir = Path(log_dir)

        log_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = log_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def _move_batch_to_device(self, batch: tuple | list | dict) -> tuple | list | dict | torch.Tensor:
        if batch is None:
            raise ValueError("Batch cannot be none")

        if isinstance(batch, (tuple, list, dict)):
            if len(batch) == 0:
                raise ValueError("Batch cannot be empty")
            if isinstance(batch, (tuple, list)):
                return [b.to(self.device) for b in batch]
            if isinstance(batch, dict):
                return {key: value.to(self.device) for key, value in batch.items()}

        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)

        raise ValueError(f"Error transferring a batch of an unsupported format ({type(batch)})")

    @staticmethod
    def _unpack_batch(batch: tuple | list | dict) -> tuple:
        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                return batch[0], batch[1]
            elif len(batch) >= 3:
                return batch[0], batch[1], batch[2]
            else:
                raise ValueError(f"Batch must contain at least 2 elements, got {len(batch)}")

        elif isinstance(batch, dict):
            if "image" in batch and "mask" in batch and "valid_mask" in batch:
                return batch["image"], batch["mask"], batch["valid_mask"]
            elif "image" in batch and "mask" in batch:
                return batch["image"], batch["mask"]
            else:
                raise ValueError(f"Dictionary must contain keys 'image' and 'mask', got {batch.keys()}")

        elif torch.is_tensor(batch):
            raise ValueError("Batch must contain both images and masks")

        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

    @torch.no_grad()
    def _update_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metrics: dict[str, torchmetrics.Metric],
        threshold: float | None = None,
    ) -> None:
        self._validate_tensor(predictions, "predictions")
        self._validate_tensor(targets, "targets")
        metrics = self._validate_metrics(metrics)

        if threshold is None:
            threshold = self.learning_config.threshold

        probs = torch.sigmoid(predictions)
        preds = (probs > threshold).long()
        for name, metric in metrics.items():
            metric.update(preds, targets.long())

    def _reset_metrics(self, metrics: dict[str, torchmetrics.Metric]) -> None:
        metrics = self._validate_metrics(metrics)
        for metric in metrics.values():
            metric.reset()

    def _compute_metrics(self, metrics: dict[str, torchmetrics.Metric]) -> dict[str, float]:
        metrics = self._validate_metrics(metrics)
        return {name: metric.compute().item() for name, metric in metrics.items()}

    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> tuple | torch.Tensor:
        self._validate_tensor(predictions, "predictions")
        self._validate_tensor(targets, "targets")

        if valid_mask is not None:
            if isinstance(self.loss_function, ComboLoss) and return_components:
                total_loss, components = self.loss_function.forward_with_components(
                    predictions,
                    targets,
                    valid_mask=valid_mask
                )
                return total_loss, components
            else:
                loss = self.loss_function(predictions, targets, valid_mask=valid_mask)
                if return_components:
                    return loss, {"loss": loss}
                return loss

        else:
            if isinstance(self.loss_function, ComboLoss) and return_components:
                total_loss, components = self.loss_function.forward_with_components(predictions, targets)
                return total_loss, components
            else:
                loss = self.loss_function(predictions, targets)
                if return_components:
                    return loss, {"loss": loss}
                return loss

    def _learn(self):
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.learning_config.max_clip_grad_norm)
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
        else:
            self.optimizer.step()
        if self.scaler.is_enabled():
            self.scaler.update()
        self.optimizer.zero_grad()

    def _learn_with_pixels_accumulation(self, accumulated_pixels):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad /= accumulated_pixels
        self._learn()

    def run_epoch(
        self,
        dataloader: data.DataLoader,
        mode: str,
        metrics: dict[str, torchmetrics.Metric] | None = None,
        resume_batches: int = 0,
    ) -> dict[str, float]:
        mode = self._validate_run_epoch_mode(mode)
        dataloader = self._validate_dataloader(dataloader)

        if self.device.type == "cuda" and torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated() / 1024**2
            self.logger.debug("Memory before %s: %.1fMB", mode, memory_before)

        if mode == "train":
            train = True
            self._request_optimizer()
            grad_context = nullcontext()
            self.model.train()
        else:
            train = False
            grad_context = torch.no_grad()
            self.model.eval()

        pixels_per_step = self.learning_config.pixels_per_step
        accumulation_steps = self.learning_config.accumulation_steps
        use_pixels_accumulation = pixels_per_step > 0
        if use_pixels_accumulation:
            accumulated_pixels = 0

        start_time = time.time()
        total_loss = 0.0
        num_batches = 0
        current_metrics = metrics if metrics is not None else self.metrics
        self._reset_metrics(current_metrics)

        if self.has_components:
            total_components = {name: 0.0 for name in self.loss_function.names}
        else:
            total_components = {"loss": 0.0}

        desc = "Evaluating..." if mode == "test" else f"{mode.capitalize()} Epoch {self.current_epoch}"
        pbar = tqdm(desc=desc, leave=True, unit="batch")

        data_iter = iter(dataloader)
        skipped = 0
        if resume_batches > 0:
            for _ in range(resume_batches):
                try:
                    next(data_iter)
                    skipped += 1
                except StopIteration:
                    break
            self.logger.debug("Skipped %d batches (requested %d)", skipped, resume_batches)

        with grad_context:
            if train:
                self.optimizer.zero_grad()
            while True:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                except Exception as ex:
                    self.logger.error("Error loading batch %d: %s, skipped", num_batches, ex)
                    continue

                num_batches += 1
                if mode == "train":
                    self.current_batch_in_epoch = num_batches

                try:
                    batch = self._move_batch_to_device(batch)
                    if len(batch) == 3:
                        x, y, valid_mask = self._unpack_batch(batch)
                    else:
                        x, y = self._unpack_batch(batch)
                        valid_mask = None

                    if valid_mask is not None:
                        num_pixels = valid_mask.sum().item()
                    else:
                        num_pixels = x.shape[0] * x.shape[2] * x.shape[3]

                    with autocast(self.device.type, enabled=self.scaler.is_enabled()):
                        predictions = self.model(x)
                        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                            self.logger.warning("NaN/Inf in model output, skipping batch %d", num_batches)
                            self.optimizer.zero_grad()
                            if use_pixels_accumulation:
                                accumulated_pixels = 0
                            continue

                        loss, components = self._compute_loss(predictions, y, valid_mask, return_components=True)

                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.warning("NaN or Inf loss detected, skipping batch %d", num_batches)
                        self.optimizer.zero_grad()
                        if use_pixels_accumulation:
                            accumulated_pixels = 0
                        continue

                    if train:
                        if use_pixels_accumulation:
                            scaled_loss = loss * num_pixels
                            self.scaler.scale(scaled_loss).backward()
                            accumulated_pixels += num_pixels
                            if accumulated_pixels >= pixels_per_step:
                                self._learn_with_pixels_accumulation(accumulated_pixels)
                                accumulated_pixels = 0

                        else:
                            loss = loss / accumulation_steps
                            self.scaler.scale(loss).backward()
                            if num_batches % accumulation_steps == 0:
                                self._learn()

                    self._update_metrics(predictions, y, current_metrics)
                    total_loss += loss.item()
                    for name in total_components.keys():
                        total_components[name] += components[name].item()

                    pbar.update(1)
                    pbar.set_postfix({
                        "loss": f"{loss.item():.5f}",
                        "speed": (
                            f"{num_batches / (time.time() - start_time):.2f} it/s" if num_batches > 0 else "N/A"
                        )})

                except Exception as ex:
                    self.logger.error(f"Error in {mode} batch {num_batches}: {ex}")
                    if mode == "train":
                        self.logger.warning("Skipping train batch %s", num_batches)
                        self.optimizer.zero_grad()
                        if use_pixels_accumulation:
                            accumulated_pixels = 0
                        continue
                    else:
                        raise

            if train and use_pixels_accumulation and accumulated_pixels > 0:
                self._learn_with_pixels_accumulation(accumulated_pixels)
            elif train and not use_pixels_accumulation and num_batches % accumulation_steps != 0:
                self._learn()

        if num_batches == 0:
            self.logger.warning("No batches processed, returning NaN metrics")
            return {
                "loss": float("nan"),
                **{k: float("nan") for k in current_metrics},
                **{k: float("nan") for k in total_components},
            }

        if self.device.type == "cuda" and torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated() / 1024**2
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            self.logger.debug(
                "%s memory - Final: %.1fMB, Peak: %.1fMB",
                mode.capitalize(), memory_after, peak_memory
            )

        elapsed_time = time.time() - start_time
        self.logger.info(
            "%s epoch completed in %.2fs, %.2f batches/sec",
            mode.capitalize(), elapsed_time, num_batches / elapsed_time
        )

        avg_loss = total_loss / num_batches
        avg_components = {name: total / num_batches for name, total in total_components.items()}
        metrics_values = self._compute_metrics(current_metrics)
        self.current_batch_in_epoch = 0

        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics_values.items()])
        self.logger.info(
            "%s completed: loss=%.4f, %s",
            mode.capitalize(), avg_loss, metrics_str
        )

        return {"loss": avg_loss, **metrics_values, **avg_components}
