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
import torchmetrics
from torch.amp import GradScaler, autocast
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
        self.loss_function = self._validate_loss_function(loss_function)
        self.config = self._validate_config(config)
        self.device = self._validate_device(device)

        if optimizer is not None:
            self.optimizer = self._validate_optimizer(optimizer)
        else:
            self.optimizer = None

        model = self._validate_model(model)
        model = model.to(self.device)
        if self.config["learning"].get("compile_model", False):
            model = torch.compile(
                model,
                dynamic=self.config["learning"].get("compile_dynamic", True),
                mode=self.config["learning"].get("compile_options", "default")
            )
        self.model = model

        use_amp = self.config["learning"].get("use_fp16", False) and self.device.type == "cuda"
        self.scaler = GradScaler(self.device.type, enabled=use_amp)
        self.log_dir = self._ensure_log_dir(log_dir)
        self.has_components = isinstance(self.loss_function, ComboLoss)
        self.current_epoch = 0
        self.current_batch_in_epoch = 0

        if metrics is None or metrics == {}:
            self.metrics = {}
        else:
            self.metrics = {name: metric.to(self.device) for name, metric in metrics.items()}
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self.config["model"].get("model_name", "model")

    def _validate_model(self, model: nn.Module) -> nn.Module:
        if model is None:
            error_msg = "model cannot be none"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        if not isinstance(model, nn.Module):
            error_msg = "Unsupported model type"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        return model

    def _validate_loss_function(self, loss_function: Union[nn.Module, Callable]) -> Union[nn.Module, Callable]:
        if loss_function is None:
            error_msg = "loss_function cannot be none"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        if not isinstance(loss_function, (nn.Module, Callable)):
            error_msg = "Unsupported loss_function type"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        return loss_function

    def _validate_config(self, config: Dict) -> Dict:
        if config is None:
            error_msg = "config cannot be none"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        if not isinstance(config, Dict):
            error_msg = "Unsupported configuration format"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        if len(config) == 0:
            error_msg = "config cannot be empty"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        return config

    def _validate_device(self, device: Union[torch.device, str, None]) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            try:
                return torch.device(device)
            except Exception as ex:
                error_msg = f"Invalid device parameter value: {device}. {ex}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        elif isinstance(device, torch.device):
            if device.type == "cuda" and not torch.cuda.is_available():
                error_msg = "GPU is not available"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            return device

        error_msg = f"Invalid device parameter value: {device}."
        self.logger.error(error_msg)
        raise ValueError(error_msg)

    def _validate_optimizer(self, optimizer: optim.Optimizer) -> optim.Optimizer:
        if optimizer is None:
            error_msg = "optimizer cannot be none"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)
        if not isinstance(optimizer, optim.Optimizer):
            error_msg = "Unsupported optimizer type"
            self.logger.exception(error_msg)
            raise ValueError(error_msg)
        return optimizer

    def _validate_dataloader(self, dataloader: data.DataLoader) -> data.DataLoader:
        if dataloader is None:
            error_msg = "dataloader cannot be none"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        if not isinstance(dataloader, data.DataLoader):
            error_msg = "Unsupported dataloader type"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        if len(dataloader.dataset) == 0:
            error_msg = "dataloader dataset cannot be empty"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        return dataloader

    def _validata_run_epoch_mode(self, mode: str) -> str:
        mode = mode.lower()
        correct_modes = ["train", "val", "test"]
        if mode not in correct_modes:
            error_msg = f"Unknown mode: {mode}. Available modes: {correct_modes}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        return mode

    def _validata_run_epoch_tqdm_mode(self, tqdm_mode: str) -> str:
        tqdm_mode = tqdm_mode.lower()
        correct_tqdm_modes = [None, "default", "no_len"]
        if tqdm_mode not in correct_tqdm_modes:
            error_msg = f"Unknown tqdm_mode: {tqdm_mode}. Available tqdm_modes: {correct_tqdm_modes}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        return tqdm_mode

    def _validate_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        if predictions is None:
            error_msg = "predictions cannot be none"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        elif not isinstance(predictions, torch.Tensor):
            error_msg = "Unsupported predictions format"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        return predictions

    def _validate_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if targets is None:
            error_msg = "targets cannot be none"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        elif not isinstance(targets, torch.Tensor):
            error_msg = "Unsupported targets format"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        return targets

    def _validate_metrics(self, metrics: Dict[str, torchmetrics.Metric]) -> Dict[str, torchmetrics.Metric]:
        if metrics is None or metrics == {}:
            error_msg = "metrics cannot be none or empty"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        if isinstance(metrics, torchmetrics.Metric):
            return {type(metrics).__name__: metrics}
        if isinstance(metrics, dict):
            return metrics
        error_msg = "Unsupported metrics format"
        self.logger.error(error_msg)
        raise ValueError(error_msg)

    def _run_epoch_request_optimizer(self):
        if self.optimizer is None:
            error_msg = "Optimizer required for training mode"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        if not isinstance(self.optimizer, optim.Optimizer):
            error_msg = "Unsupported optimizer type"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _ensure_log_dir(self, log_dir: Union[str, Path]) -> Path:
        if not isinstance(log_dir, (str, Path)) and log_dir is not None:
            error_msg = "unsupported path format"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if log_dir is None:
            log_dir = Path.cwd() / "logs"
            self.logger.warning(f"Entered log_dir is empty. Auto log_dir: {log_dir}")
        else:
            log_dir = Path(log_dir)

        log_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = log_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def _move_batch_to_device(self, batch: Union[Tuple, List, Dict]) -> Union[Tuple, List, Dict, torch.Tensor]:
        if batch is None:
            error_msg = "Batch cannot be none"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        if isinstance(batch, (Tuple, List, Dict)):
            if len(batch) == 0:
                error_msg = "Batch cannot be empty"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            if isinstance(batch, (Tuple, List)):
                return [b.to(self.device) for b in batch]
            if isinstance(batch, Dict):
                return {key: value.to(self.device) for key, value in batch.items()}
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        error_msg = f"Error transferring a batch of an unsupported format ({type(batch)})"
        self.logger.error(error_msg)
        raise ValueError(error_msg)

    def _unpack_batch(self, batch: Union[Tuple, List, Dict]) -> Tuple:
        if isinstance(batch, (Tuple, List)):
            if len(batch) == 2:
                return batch[0], batch[1]
            elif len(batch) >= 3:
                return batch[0], batch[1], batch[2]
            else:
                error_msg = f"Batch must contain at least 2 elements, got {len(batch)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        elif isinstance(batch, Dict):
            if "image" in batch and "mask" in batch and "valid_mask" in batch:
                return batch["image"], batch["mask"], batch["valid_mask"]
            elif "image" in batch and "mask" in batch:
                return batch["image"], batch["mask"]
            else:
                error_msg = f"Dictionary must contain keys 'image' and 'mask', got {batch.keys()}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        elif torch.is_tensor(batch):
            error_msg = "Batch must contain both images and masks"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        else:
            error_msg = f"Unsupported batch type: {type(batch)}"
            self.logger.error(error_msg)
            raise TypeError(error_msg)

    @torch.no_grad()
    def _update_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metrics: Dict[str, torchmetrics.Metric],
        threshold: Optional[float] = None,
    ):
        self._validate_predictions(predictions)
        self._validate_targets(targets)
        metrics = self._validate_metrics(metrics)

        if not isinstance(threshold, float):
            try:
                threshold = float(threshold)
            except Exception:
                threshold = self.config["learning"]["threshold"]

        probs = torch.sigmoid(predictions)
        preds = (probs > threshold).long()
        for name, metric in metrics.items():
            metric.update(preds, targets.long())

    def _reset_metrics(self, metrics: Dict[str, torchmetrics.Metric]):
        metrics = self._validate_metrics(metrics)
        for metric in metrics.values():
            metric.reset()

    def _compute_metrics(self, metrics: Dict[str, torchmetrics.Metric]) -> Dict[str, torchmetrics.Metric]:
        metrics = self._validate_metrics(metrics)
        return {name: metric.compute().item() for name, metric in metrics.items()}

    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, return_components: bool = False
        ) -> Union[Tuple, torch.Tensor]:
        self._validate_predictions(predictions)
        self._validate_targets(targets)

        if hasattr(self.loss_function, "forward_with_components") and return_components:
            total_loss, components = self.loss_function.forward_with_components(predictions, targets)
            return total_loss, components
        else:
            loss = self.loss_function(predictions, targets)
            if return_components:
                return loss, {"loss": loss}
            return loss

    def _learn(self):
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
        else:
            self.optimizer.step()
        if self.scaler.is_enabled():
            self.scaler.update()
        self.optimizer.zero_grad()

    def run_epoch(
        self,
        dataloader: data.DataLoader,
        mode: str,
        metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        tqdm_mode: Optional[str] = "default",
        resume_batches: int = 0,
    ) -> Dict[str, float]:
        mode = self._validata_run_epoch_mode(mode)
        tqdm_mode = self._validata_run_epoch_tqdm_mode(tqdm_mode)
        self._validate_dataloader(dataloader)
        if mode == "train":
            self._run_epoch_request_optimizer()

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            memory_before = torch.cuda.memory_allocated() / 1024**2
            self.logger.debug(f"Memory before {mode}: {memory_before:.1f}MB")

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
        current_metrics = (self.metrics if metrics is None or mode in ["train", "val"]
                           else self._validate_metrics(metrics))
        self._reset_metrics(current_metrics)

        pixels_per_step = int(self.config["learning"].get("pixels_per_step", 0))
        accumulation_steps = self.config["learning"].get("accumulation_steps", 1)
        use_pixels_accumulation = pixels_per_step > 0
        if pixels_per_step:
            accumulated_pixels = 0

        if self.has_components:
            total_components = {name: 0.0 for name in self.loss_function.names}
        else:
            total_components = {"loss": 0.0}
        desc = "Evaluating..." if mode == "test" else f"{mode.capitalize()} Epoch {self.current_epoch}"

        data_iter = iter(dataloader)
        if resume_batches > 0:
            for _ in range(resume_batches):
                next(data_iter)

        if tqdm_mode is not None:
            if tqdm_mode == "default":
                data_iter = tqdm(data_iter, desc=desc, leave=True)
            elif tqdm_mode == "no_len":

                class NoLenIterable:
                    def __init__(self, iterable):
                        self.iterable = iterable

                    def __iter__(self):
                        return iter(self.iterable)

                data_iter = tqdm(NoLenIterable(data_iter), desc=desc, leave=True)

        with grad_context:
            if train:
                self.optimizer.zero_grad()
            for batch in data_iter:
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
                        loss, components = self._compute_loss(predictions, y, return_components=True)

                    if train:
                        if use_pixels_accumulation:
                            scaled_loss = loss * num_pixels
                            self.scaler.scale(scaled_loss).backward()
                            accumulated_pixels += num_pixels
                            if accumulated_pixels >= pixels_per_step:
                                for param in self.model.parameters():
                                    if param.grad is not None:
                                        param.grad /= accumulated_pixels
                                self._learn()
                                accumulated_pixels = 0
                        else:
                            loss = loss / accumulation_steps
                            self.scaler.scale(loss).backward()
                            if (num_batches + 1) % accumulation_steps == 0:
                                self._learn()

                    self._update_metrics(predictions, y, current_metrics)
                    total_loss += loss.item()
                    num_batches += 1
                    self.current_batch_in_epoch = num_batches
                    for name in total_components.keys():
                        total_components[name] += components[name].item()

                    if isinstance(data_iter, tqdm):
                        data_iter.set_postfix({
                            "loss": f"{loss.item():.4f}",
                            "speed": (
                                f"{num_batches / (time.time() - start_time):.1f} it/s" if num_batches > 0 else "N/A"
                            )})
                except Exception as ex:
                    self.logger.error(f"Error in {mode} batch {num_batches}: {ex}")
                    if mode == "train":
                        self.logger.warning(f"Skipping train batch {num_batches}")
                        if "x" in locals():
                            del x
                        if "y" in locals():
                            del y
                        if "predictions" in locals():
                            del predictions
                        if "loss" in locals():
                            del loss
                        if "components" in locals():
                            del components
                        if "valid_mask" in locals():
                            del valid_mask
                        torch.cuda.empty_cache()
                        self.optimizer.zero_grad()
                        if use_pixels_accumulation:
                            accumulated_pixels = 0
                        continue
                    else:
                        raise

            if train and use_pixels_accumulation and accumulated_pixels > 0:
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad /= accumulated_pixels
                self._learn()
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
                f"{mode.capitalize()} memory - Final: {memory_after:.1f}MB, " f"Peak: {peak_memory:.1f}MB"
            )

        elapsed_time = time.time() - start_time
        self.logger.info(
            f"{mode.capitalize()} epoch completed in {elapsed_time:.2f}s, "
            f"{num_batches / elapsed_time:.2f} batches/sec"
        )

        avg_loss = total_loss / num_batches
        avg_components = {name: total / num_batches for name, total in total_components.items()}
        metrics_values = self._compute_metrics(current_metrics)
        self.current_batch_in_epoch = 0

        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics_values.items()])
        self.logger.info(f"{mode.capitalize()} completed: loss={avg_loss:.4f}, {metrics_str}")

        return {"loss": avg_loss, **metrics_values, **avg_components}
