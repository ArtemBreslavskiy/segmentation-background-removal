import pytest
import torch
import torch.nn as nn
import torch.utils.data as data
import torchmetrics
import logging
from torch.cuda.amp import GradScaler
from typing import Callable
from copy import deepcopy
from unittest.mock import Mock

from src.engine.BaseModule import BaseModule
from src.losses.ComboLoss import ComboLoss
from tests.conftest import batch_generator


class TestBaseModule:
    def test_init(
        self,
        dummy_model,
        dummy_config,
        dummy_loss_function,
        dummy_optimizer,
        tmp_path,
        dummy_metrics,
        dummy_logger
    ):
        log_dir = tmp_path / "test_dir"
        base = BaseModule(
            model=dummy_model,
            config=dummy_config,
            loss_function=dummy_loss_function,
            optimizer=dummy_optimizer,
            log_dir=log_dir,
            metrics=dummy_metrics,
            device="cpu",
            logger=dummy_logger,
            model_name="test_model"
        )

        assert base.model == dummy_model
        assert base.config == dummy_config
        assert base.loss_function == dummy_loss_function
        assert base.optimizer == dummy_optimizer
        assert base.log_dir == log_dir
        assert base.metrics == dummy_metrics
        assert base.logger == dummy_logger
        assert base.model_name == "test_model"

        assert base.current_epoch == 0
        assert base.has_components == False

        assert isinstance(base.loss_function, Callable)
        assert isinstance(base.scaler, GradScaler)

        assert base.log_dir.exists()
        assert base.log_dir.is_dir()
        assert (base.log_dir / "checkpoints").exists()
        assert (base.log_dir / "checkpoints").is_dir()

    def test_init_with_combo_loss(
        self,
        dummy_model,
        dummy_config,
        dummy_combo_loss_function,
        dummy_optimizer,
        tmp_path,
        dummy_metrics,
        dummy_logger
    ):
        log_dir = tmp_path / "test_dir"
        base = BaseModule(
            model=dummy_model,
            config=dummy_config,
            loss_function=dummy_combo_loss_function,
            optimizer=dummy_optimizer,
            log_dir=log_dir,
            metrics=dummy_metrics,
            device='cpu',
            logger=dummy_logger,
            model_name="test_model"
        )

        assert base.loss_function == dummy_combo_loss_function
        assert base.has_components == True

    def test_init_without_model_name(
        self,
        dummy_model,
        dummy_config,
        dummy_loss_function,
        tmp_path,
        dummy_logger
    ):
        log_dir = tmp_path / "test_dir"
        base = BaseModule(
            model=dummy_model,
            config=dummy_config,
            loss_function=dummy_loss_function,
            log_dir=log_dir,
            device="cpu",
            logger=dummy_logger,
        )
        assert base.model_name == 'TestModel'

    def test_init_without_logger(
        self,
        dummy_model,
        dummy_config,
        dummy_loss_function,
        tmp_path,
        dummy_metrics,
    ):
        log_dir = tmp_path / "test_dir"
        base = BaseModule(
            model=dummy_model,
            config=dummy_config,
            loss_function=dummy_loss_function,
            log_dir=log_dir,
            device="cpu",
        )
        assert isinstance(base.logger, logging.Logger)

    def test_init_without_model(
        self,
        caplog,
        dummy_config,
        dummy_loss_function,
        tmp_path,
        dummy_logger
    ):
        log_dir = tmp_path / "test_dir"

        with pytest.raises(ValueError, match="model cannot be none"):
            BaseModule(
                model=None,
                config=dummy_config,
                loss_function=dummy_loss_function,
                log_dir=log_dir,
                device="cpu",
                logger=dummy_logger
            )
        assert any(
            rec.levelname == "ERROR" and "model cannot be none" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("model", [10, 0.5, "test"])
    def test_init_with_unsupported_model_type(
        self,
        caplog,
        dummy_config,
        dummy_loss_function,
        tmp_path,
        dummy_logger,
        model
    ):
        log_dir = tmp_path / "test_dir"

        with pytest.raises(ValueError, match="Unsupported model type"):
            BaseModule(
                model=model,
                config=dummy_config,
                loss_function=dummy_loss_function,
                log_dir=log_dir,
                device="cpu",
                logger=dummy_logger
            )
        assert any(
            rec.levelname == "ERROR" and "Unsupported model type" in rec.message
            for rec in caplog.records
        )

    def test_init_without_loss_function(
        self,
        caplog,
        dummy_model,
        dummy_config,
        tmp_path,
        dummy_logger
    ):
        log_dir = tmp_path / "test_dir"

        with pytest.raises(ValueError, match="loss_function cannot be none"):
            BaseModule(
                model=dummy_model,
                config=dummy_config,
                loss_function=None,
                log_dir=log_dir,
                device="cpu",
                logger=dummy_logger
            )
        assert any(
            rec.levelname == "ERROR" and "loss_function cannot be none" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("loss_function", [10, 0.5, "test"])
    def test_init_with_unsupported_loss_function_type(
        self,
        caplog,
        dummy_model,
        dummy_config,
        tmp_path,
        dummy_logger,
        loss_function
    ):
        log_dir = tmp_path / "test_dir"

        with pytest.raises(ValueError, match="Unsupported loss_function type"):
            BaseModule(
                model=dummy_model,
                config=dummy_config,
                loss_function=loss_function,
                log_dir=log_dir,
                device="cpu",
                logger=dummy_logger
            )
        assert any(
            rec.levelname == "ERROR" and "Unsupported loss_function type" in rec.message
            for rec in caplog.records
        )

    def test_init_without_config(
        self,
        caplog,
        dummy_model,
        dummy_loss_function,
        tmp_path,
        dummy_logger
    ):
        log_dir = tmp_path / "test_dir"

        with pytest.raises(ValueError, match="config cannot be none"):
            BaseModule(
                model=dummy_model,
                config=None,
                loss_function=dummy_loss_function,
                log_dir=log_dir,
                device="cpu",
                logger=dummy_logger
            )
        assert any(
            rec.levelname == "ERROR" and "config cannot be none" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("config", [10, 0.5, "test"])
    def test_init_with_unsupported_config_type(
        self,
        caplog,
        dummy_model,
        dummy_loss_function,
        tmp_path,
        dummy_logger,
        config
    ):
        log_dir = tmp_path / "test_dir"

        with pytest.raises(ValueError, match="Unsupported config type"):
            BaseModule(
                model=dummy_model,
                config=config,
                loss_function=dummy_loss_function,
                log_dir=log_dir,
                device="cpu",
                logger=dummy_logger
            )
        assert any(
            rec.levelname == "ERROR" and "Unsupported config type" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("device", [10, 0.5, "test"])
    def test_init_with_invalid_device(
        self,
        caplog,
        dummy_model,
        dummy_config,
        dummy_loss_function,
        tmp_path,
        dummy_logger,
        device
    ):
        log_dir = tmp_path / "test_dir"

        with pytest.raises(ValueError, match="Invalid device parameter value"):
            BaseModule(
                model=dummy_model,
                config=dummy_config,
                loss_function=dummy_loss_function,
                log_dir=log_dir,
                device=device,
                logger=dummy_logger
            )
        assert any(
            rec.levelname == "ERROR" and "Invalid device parameter value" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("device", ["cuda", torch.device("cuda")])
    def test_init_with_cuda_when_unavailable(
        self,
        caplog,
        monkeypatch,
        dummy_model,
        dummy_config,
        dummy_loss_function,
        tmp_path,
        device,
        dummy_logger
    ):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        log_dir = tmp_path / "test_dir"
        with pytest.raises(ValueError, match="GPU is not available"):
            BaseModule(
                model=dummy_model,
                config=dummy_config,
                loss_function=dummy_loss_function,
                log_dir=log_dir,
                device=device,
                logger=dummy_logger
            )
        assert any(
            rec.levelname == "ERROR" and "GPU is not available" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda", torch.device("cpu"), torch.device("cuda")])
    def test_model_location(
        self,
        dummy_model,
        dummy_config,
        dummy_loss_function,
        tmp_path,
        device,
        dummy_logger
    ):
        if "cuda" in str(device) and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        log_dir = tmp_path / "test_dir"
        base = BaseModule(
            model=dummy_model,
            config=dummy_config,
            loss_function=dummy_loss_function,
            log_dir=log_dir,
            device=device,
            logger=dummy_logger
        )
        expected_type = device.type if isinstance(device, torch.device) else device
        actual_type = next(base.model.parameters()).device.type
        assert expected_type == actual_type

    @pytest.mark.parametrize("batch_length", [2, 5, 7])
    @pytest.mark.parametrize("batch_type", ["tuple", "list", "dict"])
    @pytest.mark.parametrize("batch_item", [1, 2, 3])
    def test_move_batch_to_device(
        self,
        batch_length,
        batch_type,
        batch_item,
        dummy_model,
        dummy_config,
        dummy_loss_function,
        tmp_path,
        dummy_logger
    ):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch = batch_generator(batch_length, batch_type, batch_item)
        log_dir = tmp_path / "test_dir"
        base = BaseModule(
            model=dummy_model,
            config=dummy_config,
            loss_function=dummy_loss_function,
            log_dir=log_dir,
            device="cuda",
            logger=dummy_logger
        )
        moved_batch = base._move_batch_to_device(batch)
        if isinstance(moved_batch, (tuple, list)):
            for b in moved_batch:
                assert b.device.type == "cuda"
        elif isinstance(moved_batch, dict):
            for _, b in moved_batch.items():
                assert b.device.type == "cuda"

    @pytest.mark.parametrize("batch", [None, []])
    def test_move_batch_to_device_with_empty_batch(self, dummy_base_module, batch, caplog):
        base = dummy_base_module
        with pytest.raises(ValueError, match="Batch cannot be"):
            base._move_batch_to_device(batch)
        assert any(
            rec.levelname == "ERROR" and "Batch cannot be" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("batch", [10, 0.5, "test"])
    def test_move_batch_to_device_with_unsupported_batch_format(self, dummy_base_module, batch, caplog):
        base = dummy_base_module
        with pytest.raises(ValueError, match="Error transferring a batch of an unsupported format"):
            base._move_batch_to_device(batch)
        assert any(
            rec.levelname == "ERROR" and "Error transferring a batch of an unsupported format" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("batch_length", [1, 2, 3])
    @pytest.mark.parametrize("batch_type", ["tuple", "list", "dict"])
    @pytest.mark.parametrize("batch_item", [2, 3])
    def test_unpack_batch(
        self,
        batch_length,
        batch_type,
        batch_item,
        dummy_base_module
    ):
        batch = batch_generator(batch_length, batch_type, batch_item)
        base = dummy_base_module
        unpacked = base._unpack_batch(batch)
        assert len(unpacked) == batch_item
        assert isinstance(unpacked, tuple)

    @pytest.mark.parametrize("batch_type", ["tuple", "list", "dict"])
    def test_unpack_batch_with_1_element(self, batch_type, dummy_base_module, caplog):
        batch = batch_generator(2, batch_type, 1)
        base = dummy_base_module
        if batch_type in ["tuple", "list"]:
            with pytest.raises(ValueError, match="Batch must contain at least 2 elements"):
                base._unpack_batch(batch)
            assert any(
                rec.levelname == "ERROR" and "Batch must contain at least 2 elements" in rec.message
                for rec in caplog.records
            )
        elif batch_type == "dict":
            with pytest.raises(ValueError, match="Dictionary must contain keys 'image' and 'mask'"):
                base._unpack_batch(batch)
            assert any(
                rec.levelname == "ERROR" and "Dictionary must contain keys 'image' and 'mask'" in rec.message
                for rec in caplog.records
            )

    def test_unpack_batch_with_tensor(self, dummy_base_module, caplog):
        batch = torch.randint(0, 255, (1, 1, 2, 2))
        base = dummy_base_module
        with pytest.raises(ValueError, match="Batch must contain both images and masks"):
            base._unpack_batch(batch)
        assert any(
            rec.levelname == "ERROR" and "Batch must contain both images and masks" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("batch", [10, 0.5, "test"])
    def test_unpack_batch_with_unsupported_batch_type(self, dummy_base_module, batch, caplog):
        base = dummy_base_module
        with pytest.raises(TypeError, match="Unsupported batch type"):
            base._unpack_batch(batch)
        assert any(
            rec.levelname == "ERROR" and "Unsupported batch type" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("threshold", [None, 0.5])
    def test_metrics(self, dummy_base_module, threshold):
        base = dummy_base_module
        preds = torch.randint(0, 2, (4, 1, 32, 32)).long()
        targets = torch.randint(0, 2, (4, 1, 32, 32)).long()

        iou_fn = torchmetrics.JaccardIndex(task="binary")
        iou_fn.update(preds, targets)
        iou = iou_fn.compute().item()
        iou_fn.reset()

        accuracy_fn = torchmetrics.Accuracy(task="binary")
        accuracy_fn.update(preds, targets)
        accuracy = accuracy_fn.compute().item()
        accuracy_fn.reset()

        base._reset_metrics(base.metrics)
        if threshold:
            base._update_metrics(preds, targets, base.metrics, threshold=threshold)
        else:
            base._update_metrics(preds, targets, base.metrics)
        computed = base._compute_metrics(base.metrics)

        assert isinstance(computed, dict)
        assert "iou" in computed
        assert "accuracy" in computed

        assert isinstance(computed["iou"], float)
        assert isinstance(computed["accuracy"], float)

        assert computed["iou"] == pytest.approx(iou, abs(1e-4))
        assert computed["accuracy"] == pytest.approx(accuracy, abs(1e-4))

        base._reset_metrics(base.metrics)
        if threshold:
            base._update_metrics(torch.zeros_like(preds), torch.ones_like(targets), base.metrics, threshold=threshold)
        else:
            base._update_metrics(torch.zeros_like(preds), torch.ones_like(targets), base.metrics)
        computed2 = base._compute_metrics(base.metrics)
        assert computed2["iou"] != computed["iou"]
        assert computed2["accuracy"] != computed["accuracy"]

    def test_update_metrics_without_preds(self, dummy_base_module, caplog):
        base = dummy_base_module
        targets = torch.randint(0, 2, (4, 1, 32, 32)).long()
        with pytest.raises(ValueError, match="predictions cannot be none"):
            base._update_metrics(predictions=None, targets=targets, metrics=base.metrics)
        assert any(
            rec.levelname == "ERROR" and "predictions cannot be none" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("preds", [10, 0.5, "test"])
    def test_update_metrics_with_unsupported_preds_format(self, dummy_base_module, preds, caplog):
        base = dummy_base_module
        targets = torch.randint(0, 2, (4, 1, 32, 32)).long()
        with pytest.raises(ValueError, match="Unsupported predictions format"):
            base._update_metrics(predictions=preds, targets=targets, metrics=base.metrics)
        assert any(
            rec.levelname == "ERROR" and "Unsupported predictions format" in rec.message
            for rec in caplog.records
        )

    def test_update_metrics_without_targets(self, dummy_base_module, caplog):
        base = dummy_base_module
        preds = torch.randint(0, 2, (4, 1, 32, 32)).long()
        with pytest.raises(ValueError, match="targets cannot be none"):
            base._update_metrics(predictions=preds, targets=None, metrics=base.metrics)
        assert any(
            rec.levelname == "ERROR" and "targets cannot be none" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("targets", [10, 0.5, "test"])
    def test_update_metrics_with_unsupported_targets_format(self, dummy_base_module, targets, caplog):
        base = dummy_base_module
        preds = torch.randint(0, 2, (4, 1, 32, 32)).long()
        with pytest.raises(ValueError, match="Unsupported targets format"):
            base._update_metrics(predictions=preds, targets=targets, metrics=base.metrics)
        assert any(
            rec.levelname == "ERROR" and "Unsupported targets format" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("metrics", [None, {}])
    def test_update_metrics_without_metrics(self, dummy_base_module, metrics, caplog):
        base = dummy_base_module
        preds = torch.randint(0, 2, (4, 1, 32, 32)).long()
        targets = torch.randint(0, 2, (4, 1, 32, 32)).long()
        with pytest.raises(ValueError, match="metrics cannot be none or empty"):
            base._update_metrics(predictions=preds, targets=targets, metrics=metrics)
        assert any(
            rec.levelname == "ERROR" and "metrics cannot be none" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("metrics", [10, 0.5, "test"])
    def test_update_metrics_with_unsupported_metrics_format(self, dummy_base_module, metrics, caplog):
        base = dummy_base_module
        preds = torch.randint(0, 2, (4, 1, 32, 32)).long()
        targets = torch.randint(0, 2, (4, 1, 32, 32)).long()
        with pytest.raises(ValueError, match="Unsupported metrics format"):
            base._update_metrics(predictions=preds, targets=targets, metrics=metrics)
        assert any(
            rec.levelname == "ERROR" and "Unsupported metrics format" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("metrics", [None, {}])
    def test_reset_metrics_without_metrics(self, dummy_base_module, metrics, caplog):
        base = dummy_base_module
        with pytest.raises(ValueError, match="metrics cannot be none or empty"):
            base._reset_metrics(metrics=metrics)
        assert any(
            rec.levelname == "ERROR" and "metrics cannot be none" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("metrics", [10, 0.5, "test"])
    def test_reset_metrics_with_unsupported_metrics_format(self, dummy_base_module, metrics, caplog):
        base = dummy_base_module
        with pytest.raises(ValueError, match="Unsupported metrics format"):
            base._reset_metrics(metrics=metrics)
        assert any(
            rec.levelname == "ERROR" and "Unsupported metrics format" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("metrics", [None, {}])
    def test_compute_metrics_without_metrics(self, dummy_base_module, metrics, caplog):
        base = dummy_base_module
        with pytest.raises(ValueError, match="metrics cannot be none or empty"):
            base._compute_metrics(metrics=metrics)
        assert any(
            rec.levelname == "ERROR" and "metrics cannot be none or empty" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("metrics", [10, 0.5, "test"])
    def test_compute_metrics_with_unsupported_metrics_format(self, dummy_base_module, metrics, caplog):
        base = dummy_base_module
        with pytest.raises(ValueError, match="Unsupported metrics format"):
            base._compute_metrics(metrics=metrics)
        assert any(
            rec.levelname == "ERROR" and "Unsupported metrics format" in rec.message
            for rec in caplog.records
        )

    def test_compute_loss(self, dummy_base_module):
        base = dummy_base_module
        preds = torch.randint(0, 2, (4, 1, 32, 32)).float()
        targets = torch.randint(0, 2, (4, 1, 32, 32)).float()
        actual_loss = base._compute_loss(preds, targets, return_components=False)
        expected_loss = nn.BCEWithLogitsLoss()(preds, targets)

        assert isinstance(expected_loss, torch.Tensor)
        assert expected_loss.shape == ()
        assert actual_loss == pytest.approx(expected_loss, abs=1e-4)

    def test_compute_loss_with_components(self, dummy_base_module):
        base = dummy_base_module
        preds = torch.randint(0, 2, (4, 1, 32, 32)).float()
        targets = torch.randint(0, 2, (4, 1, 32, 32)).float()
        actual_loss, actual_components = base._compute_loss(preds, targets, return_components=True)
        expected_loss = nn.BCEWithLogitsLoss()(preds, targets)

        assert isinstance(expected_loss, torch.Tensor)
        assert isinstance(actual_components, dict)
        assert expected_loss.shape == ()
        assert actual_loss == pytest.approx(expected_loss, abs=1e-4)
        assert actual_components["loss"] == pytest.approx(expected_loss, abs=1e-4)

    def test_compute_loss_with_combo_loss_no_components(
        self,
        dummy_model,
        dummy_config,
        dummy_combo_loss_function,
        tmp_path,
        dummy_logger
    ):
        log_dir = tmp_path / "test_dir"
        base = BaseModule(
            model=dummy_model,
            config=dummy_config,
            loss_function=dummy_combo_loss_function,
            log_dir=log_dir,
            device="cpu",
            logger=dummy_logger
        )
        preds = torch.randint(0, 2, (4, 1, 32, 32)).float()
        targets = torch.randint(0, 2, (4, 1, 32, 32)).float()
        actual_loss = base._compute_loss(preds, targets, return_components=False)
        expected_loss = nn.BCEWithLogitsLoss()(preds, targets) * 0.3 + nn.MSELoss()(preds, targets) * 0.7

        assert isinstance(base.loss_function, ComboLoss)
        assert isinstance(expected_loss, torch.Tensor)
        assert expected_loss.shape == ()
        assert actual_loss == pytest.approx(expected_loss, abs=1e-4)

    def test_compute_loss_with_combo_loss_with_components(
        self,
        dummy_model,
        dummy_config,
        dummy_combo_loss_function,
        tmp_path,
        dummy_logger
    ):
        log_dir = tmp_path / "test_dir"
        base = BaseModule(
            model=dummy_model,
            config=dummy_config,
            loss_function=dummy_combo_loss_function,
            log_dir=log_dir,
            device="cpu",
            logger=dummy_logger
        )
        preds = torch.randint(0, 2, (4, 1, 32, 32)).float()
        targets = torch.randint(0, 2, (4, 1, 32, 32)).float()
        actual_loss, actual_components = base._compute_loss(preds, targets, return_components=True)
        component1 = nn.BCEWithLogitsLoss()(preds, targets)
        component2 = nn.MSELoss()(preds, targets)
        expected_loss = component1 * 0.3 + component2 * 0.7

        assert isinstance(base.loss_function, ComboLoss)
        assert isinstance(expected_loss, torch.Tensor)
        assert isinstance(actual_components, dict)
        assert expected_loss.shape == ()
        assert actual_loss == pytest.approx(expected_loss, abs=1e-4)
        assert actual_components["BCEWithLogitsLoss"] == pytest.approx(component1, abs=1e-4)
        assert actual_components["MSELoss"] == pytest.approx(component2, abs=1e-4)

    def test_compute_loss_without_preds(self, dummy_base_module, caplog):
        base = dummy_base_module
        targets = torch.randint(0, 2, (4, 1, 32, 32)).long()
        with pytest.raises(ValueError, match="predictions cannot be none"):
            base._compute_loss(predictions=None, targets=targets)
        assert any(
            rec.levelname == "ERROR" and "predictions cannot be none" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("preds", [10, 0.5, "test"])
    def test_compute_loss_with_unsupported_preds_format(self, dummy_base_module, preds, caplog):
        base = dummy_base_module
        targets = torch.randint(0, 2, (4, 1, 32, 32)).float()
        with pytest.raises(ValueError, match="Unsupported predictions format"):
            base._compute_loss(predictions=preds, targets=targets)
        assert any(
            rec.levelname == "ERROR" and "Unsupported predictions format" in rec.message
            for rec in caplog.records
        )

    def test_compute_loss_without_targets(self, dummy_base_module, caplog):
        base = dummy_base_module
        preds = torch.randint(0, 2, (4, 1, 32, 32)).float()
        with pytest.raises(ValueError, match="targets cannot be none"):
            base._compute_loss(predictions=preds, targets=None)
        assert any(
            rec.levelname == "ERROR" and "targets cannot be none" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("targets", [10, 0.5, "test"])
    def test_compute_loss_with_unsupported_targets_format(self, dummy_base_module, targets, caplog):
        base = dummy_base_module
        preds = torch.randint(0, 2, (4, 1, 32, 32)).float()
        with pytest.raises(ValueError, match="Unsupported targets format"):
            base._compute_loss(predictions=preds, targets=targets)
        assert any(
            rec.levelname == "ERROR" and "Unsupported targets format" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("mode", ["train", "val", "test"])
    def test_run_epoch(self, dummy_base_module, dummy_dataloader, mode, caplog):
        base = dummy_base_module
        metrics = base.run_epoch(dummy_dataloader, mode)

        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "iou" in metrics
        assert "accuracy" in metrics
        assert isinstance(metrics["loss"], float)

        assert any(
            rec.levelname == "INFO" and "epoch completed in" in rec.message
            for rec in caplog.records
        )
        assert any(
            rec.levelname == "INFO" and "completed: loss=" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("mode", ["train", "val", "test"])
    def test_run_epoch_with_combo_loss(
        self,
        dummy_dataloader,
        dummy_model,
        dummy_config,
        dummy_combo_loss_function,
        dummy_optimizer,
        dummy_metrics,
        tmp_path,
        dummy_logger,
        mode,
        caplog
    ):
        log_dir = tmp_path / "test_dir"
        base = BaseModule(
            model=dummy_model,
            config=dummy_config,
            loss_function=dummy_combo_loss_function,
            optimizer=dummy_optimizer,
            metrics=dummy_metrics,
            log_dir=log_dir,
            device="cpu",
            logger=dummy_logger
        )
        metrics = base.run_epoch(dummy_dataloader, mode)

        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "BCEWithLogitsLoss" in metrics
        assert "MSELoss" in metrics
        assert "iou" in metrics
        assert "accuracy" in metrics
        assert isinstance(metrics["loss"], float)

    @pytest.mark.parametrize("mode", ["train", "val", "test"])
    def test_run_epoch_cuda_logs(
        self,
        dummy_dataloader,
        dummy_model,
        dummy_config,
        dummy_loss_function,
        dummy_optimizer,
        dummy_metrics,
        tmp_path,
        dummy_logger,
        mode,
        caplog
    ):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        log_dir = tmp_path / "test_dir"
        base = BaseModule(
            model=dummy_model,
            config=dummy_config,
            loss_function=dummy_loss_function,
            optimizer=dummy_optimizer,
            metrics=dummy_metrics,
            log_dir=log_dir,
            device="cuda",
            logger=dummy_logger
        )
        metrics = base.run_epoch(dummy_dataloader, mode)

        assert isinstance(metrics, dict)
        assert metrics["loss"]
        assert any(
            rec.levelname == "INFO" and "epoch completed in" in rec.message
            for rec in caplog.records
        )
        assert any(
            rec.levelname == "INFO" and "completed: loss=" in rec.message
            for rec in caplog.records
        )
        assert any(
            rec.levelname == "DEBUG" and "Memory before" in rec.message
            for rec in caplog.records
        )
        assert any(
            rec.levelname == "DEBUG" and "memory - Final" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("mode", ["train", "val", "test"])
    def test_run_epoch_model_mode_switching(self, dummy_base_module, dummy_dataloader, mode):
        base = dummy_base_module
        if mode == "train":
            base.model.eval()
        else:
            base.model.train()
        metrics = base.run_epoch(dummy_dataloader, mode)
        if mode == "train":
            assert base.model.training
        else:
            assert not base.model.training

    @pytest.mark.parametrize("mode", ["train", "val", "test"])
    def test_run_epoch_with_metrics(self, dummy_base_module, dummy_dataloader, mode):
        base = dummy_base_module
        other_metrics = {
            "f1": torchmetrics.F1Score(task="binary"),
            "recall": torchmetrics.Recall(task="binary")
        }
        metrics = base.run_epoch(dummy_dataloader, mode, other_metrics)

        if mode == "test":
            assert "f1" in metrics
            assert "recall" in metrics
            assert "iou" not in metrics
            assert "accuracy" not in metrics
        else:
            assert "iou" in metrics
            assert "accuracy" in metrics
            assert "f1" not in metrics
            assert "recall" not in metrics
        assert "loss" in metrics

    def test_run_epoch_with_invalid_mode(self, dummy_base_module, dummy_dataloader, caplog):
        base = dummy_base_module
        with pytest.raises(ValueError, match="Unknown mode"):
            base.run_epoch(dummy_dataloader, mode="invalid")
        assert any(
            rec.levelname == "ERROR" and "Unknown mode" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("mode", ["train", "val", "test"])
    def test_run_epoch_without_optimizer(
        self,
        dummy_dataloader,
        dummy_model,
        dummy_config,
        dummy_loss_function,
        tmp_path,
        dummy_logger,
        mode,
        caplog
    ):
        log_dir = tmp_path / "test_dir"
        base = BaseModule(
            model=dummy_model,
            config=dummy_config,
            loss_function=dummy_loss_function,
            optimizer=None,
            log_dir=log_dir,
            device="cpu",
            logger=dummy_logger
        )
        if mode == "train":
            with pytest.raises(RuntimeError, match="Optimizer required for training mode"):
                base.run_epoch(dummy_dataloader, mode)
            assert any(
                rec.levelname == "ERROR" and "Optimizer required for training mode" in rec.message
                for rec in caplog.records
            )

    @pytest.mark.parametrize("mode", ["train", "val", "test"])
    @pytest.mark.parametrize("optimizer", [10, 0.5, "test"])
    def test_run_epoch_with_unsupported_optimizer_type(
        self,
        dummy_dataloader,
        dummy_model,
        dummy_config,
        dummy_loss_function,
        tmp_path,
        dummy_logger,
        mode,
        optimizer,
        caplog
    ):
        log_dir = tmp_path / "test_dir"
        base = BaseModule(
            model=dummy_model,
            config=dummy_config,
            loss_function=dummy_loss_function,
            optimizer=optimizer,
            log_dir=log_dir,
            device="cpu",
            logger=dummy_logger
        )
        if mode == "train":
            with pytest.raises(RuntimeError, match="Unsupported optimizer type"):
                base.run_epoch(dummy_dataloader, mode)
            assert any(
                rec.levelname == "ERROR" and "Unsupported optimizer type" in rec.message
                for rec in caplog.records
            )

    @pytest.mark.parametrize("mode", ["train", "val", "test"])
    def test_run_epoch_without_dataloader(self, dummy_base_module, mode, caplog):
        base = dummy_base_module
        with pytest.raises(ValueError, match="dataloader cannot be none"):
            base.run_epoch(dataloader=None, mode=mode)
        assert any(
            rec.levelname == "ERROR" and "dataloader cannot be none" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("mode", ["train", "val", "test"])
    @pytest.mark.parametrize("dataloader", [10, 0.5, "test"])
    def test_run_epoch_with_unsupported_dataloader_type(self, dummy_base_module, dataloader, mode, caplog):
        base = dummy_base_module
        with pytest.raises(ValueError, match="Unsupported dataloader type"):
            base.run_epoch(dataloader=dataloader, mode=mode)
        assert any(
            rec.levelname == "ERROR" and "Unsupported dataloader type" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("mode", ["train", "val", "test"])
    def test_run_epoch_with_empty_dataloader(self, dummy_base_module, mode, caplog):
        base = dummy_base_module
        empty_dataset = data.TensorDataset(torch.empty(0, 3, 32, 32), torch.empty(0, 1, 32, 32))
        empty_loader = data.DataLoader(empty_dataset, batch_size=2)
        with pytest.raises(ValueError, match="dataloader cannot be empty"):
            base.run_epoch(dataloader=empty_loader, mode=mode)
        assert any(
            rec.levelname == "ERROR" and "dataloader cannot be empty" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("mode", ["train", "val", "test"])
    def test_run_epoch_broken_batch(self, dummy_base_module, mode, caplog, monkeypatch):
        base = dummy_base_module
        dataset = data.TensorDataset(
            torch.randn(6, 3, 32, 32),
            torch.randint(0, 2, (6, 1, 32, 32)).float()
        )
        dataloader = data.DataLoader(dataset, batch_size=2)

        original_move = base._move_batch_to_device
        call_count = 0

        def mock_move(batch):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated data error")
            return original_move(batch)

        monkeypatch.setattr(base, "_move_batch_to_device", mock_move)

        if mode == "train":
            base.run_epoch(dataloader, mode=mode)
            assert any(
                rec.levelname == "WARNING" and "Skipping train batch" in rec.message
                for rec in caplog.records
            )
        else:
            with pytest.raises(RuntimeError, match="Simulated data error"):
                base.run_epoch(dataloader, mode=mode)
            assert any(
                rec.levelname == "ERROR" and f"Error in {mode} batch" in rec.message
                for rec in caplog.records
            )

    def test_gradient_accumulation_steps(self, dummy_base_module, dummy_config, dummy_dataloader, monkeypatch):
        base = dummy_base_module
        config = deepcopy(dummy_config)
        config["learning"]["accumulation_steps"] = 2
        config["learning"]["pixels_per_step"] = 0
        base.config = config

        mock_step = Mock()
        monkeypatch.setattr(base.optimizer, "step", mock_step)
        base.run_epoch(dummy_dataloader, "train")
        assert mock_step.call_count == 2
