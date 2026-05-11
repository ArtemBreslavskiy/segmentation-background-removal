from typing import Callable

import pytest
import torch
from torch.amp import GradScaler

from src.engine.Trainer import Trainer


class TestTrainer:
    def test_init(
        self,
        dummy_model,
        dummy_config,
        dummy_loss_function,
        dummy_optimizer,
        tmp_path,
        dummy_metrics,
        dummy_scheduler,
        dummy_logger,
    ):
        log_dir = tmp_path / "test_dir"
        trainer = Trainer(
            model=dummy_model,
            config=dummy_config,
            loss_function=dummy_loss_function,
            optimizer=dummy_optimizer,
            log_dir=log_dir,
            metrics=dummy_metrics,
            scheduler=dummy_scheduler,
            device="cpu",
            logger=dummy_logger,
            model_name="test_model",
        )

        assert trainer.model == dummy_model
        assert trainer.config == dummy_config
        assert trainer.loss_function == dummy_loss_function
        assert trainer.optimizer == dummy_optimizer
        assert trainer.log_dir == log_dir
        assert trainer.metrics == dummy_metrics
        assert trainer.scheduler == dummy_scheduler
        assert trainer.logger == dummy_logger
        assert trainer.model_name == "test_model"

        assert trainer.current_epoch == 0
        assert trainer.has_components == False

        assert isinstance(trainer.loss_function, Callable)
        assert isinstance(trainer.scaler, GradScaler)

        assert trainer.log_dir.exists()
        assert trainer.log_dir.is_dir()
        assert (trainer.log_dir / "checkpoints").exists()
        assert (trainer.log_dir / "checkpoints").is_dir()

    def test_init_without_optimizer(
        self,
        dummy_model,
        dummy_config,
        dummy_loss_function,
        tmp_path,
        dummy_metrics,
        dummy_scheduler,
        dummy_logger,
        caplog,
    ):
        log_dir = tmp_path / "test_dir"
        with pytest.raises(ValueError, match="optimizer cannot be none"):
            Trainer(
                model=dummy_model,
                config=dummy_config,
                loss_function=dummy_loss_function,
                optimizer=None,
                log_dir=log_dir,
                metrics=dummy_metrics,
                scheduler=dummy_scheduler,
                device="cpu",
                logger=dummy_logger,
                model_name="test_model",
            )
        assert any(rec.levelname == "ERROR" and "optimizer cannot be none" in rec.message for rec in caplog.records)

    @pytest.mark.parametrize("optimizer", [10, 0.5, "test"])
    def test_init_with_unsupported_optimizer_type(
        self,
        dummy_model,
        dummy_config,
        dummy_loss_function,
        tmp_path,
        dummy_metrics,
        dummy_scheduler,
        dummy_logger,
        optimizer,
        caplog,
    ):
        log_dir = tmp_path / "test_dir"
        with pytest.raises(ValueError, match="Unsupported optimizer type"):
            Trainer(
                model=dummy_model,
                config=dummy_config,
                loss_function=dummy_loss_function,
                optimizer=optimizer,
                log_dir=log_dir,
                metrics=dummy_metrics,
                scheduler=dummy_scheduler,
                device="cpu",
                logger=dummy_logger,
                model_name="test_model",
            )
        assert any(rec.levelname == "ERROR" and "Unsupported optimizer type" in rec.message for rec in caplog.records)

    @pytest.mark.parametrize("scheduler", [10, 0.5, "test"])
    def test_init_with_unsupported_scheduler_type(
        self,
        dummy_model,
        dummy_config,
        dummy_loss_function,
        dummy_optimizer,
        tmp_path,
        dummy_metrics,
        dummy_scheduler,
        dummy_logger,
        scheduler,
        caplog,
    ):
        log_dir = tmp_path / "test_dir"
        with pytest.raises(ValueError, match="Unsupported scheduler type"):
            Trainer(
                model=dummy_model,
                config=dummy_config,
                loss_function=dummy_loss_function,
                optimizer=dummy_optimizer,
                log_dir=log_dir,
                metrics=dummy_metrics,
                scheduler=scheduler,
                device="cpu",
                logger=dummy_logger,
                model_name="test_model",
            )
        assert any(rec.levelname == "ERROR" and "Unsupported scheduler type" in rec.message for rec in caplog.records)

    def test_train_epoch_without_dataloader(self, dummy_trainer, caplog):
        with pytest.raises(ValueError, match="train_dataloader cannot be none"):
            dummy_trainer.train_epoch(dataloader=None)
        assert any(
            rec.levelname == "ERROR" and "train_dataloader cannot be none" in rec.message for rec in caplog.records
        )

    @pytest.mark.parametrize("dataloader", [10, 0.5, "test"])
    def test_train_epoch_with_unsupported_dataloader_type(self, dummy_trainer, dataloader, caplog):
        with pytest.raises(ValueError, match="Unsupported train_dataloader type"):
            dummy_trainer.train_epoch(dataloader=dataloader)
        assert any(
            rec.levelname == "ERROR" and "Unsupported train_dataloader type" in rec.message for rec in caplog.records
        )

    def test_train_epoch_with_empty_dataloader(self, dummy_trainer, dummy_empty_dataloader, caplog):
        with pytest.raises(ValueError, match="train_dataloader dataset cannot be empty"):
            dummy_trainer.train_epoch(dataloader=dummy_empty_dataloader)
        assert any(
            rec.levelname == "ERROR" and "train_dataloader dataset cannot be empty" in rec.message
            for rec in caplog.records
        )

    def test_validate_epoch_without_dataloader(self, dummy_trainer, caplog):
        with pytest.raises(ValueError, match="val_dataloader cannot be none"):
            dummy_trainer.validate_epoch(dataloader=None)
        assert any(
            rec.levelname == "ERROR" and "val_dataloader cannot be none" in rec.message for rec in caplog.records
        )

    @pytest.mark.parametrize("dataloader", [10, 0.5, "test"])
    def test_validate_epoch_with_unsupported_dataloader_type(self, dummy_trainer, dataloader, caplog):
        with pytest.raises(ValueError, match="Unsupported val_dataloader type"):
            dummy_trainer.validate_epoch(dataloader=dataloader)
        assert any(
            rec.levelname == "ERROR" and "Unsupported val_dataloader type" in rec.message for rec in caplog.records
        )

    def test_validate_epoch_with_empty_dataloader(self, dummy_trainer, dummy_empty_dataloader, caplog):
        with pytest.raises(ValueError, match="val_dataloader dataset cannot be empty"):
            dummy_trainer.validate_epoch(dataloader=dummy_empty_dataloader)
        assert any(
            rec.levelname == "ERROR" and "val_dataloader dataset cannot be empty" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("log_interval", [0, -2, -5])
    def test_fit_with_invalid_log_interval(self, dummy_trainer, dummy_dataloader, log_interval, monkeypatch, caplog):
        trainer = dummy_trainer
        with pytest.raises(ValueError, match="log_interval cannot be less than 1"):
            trainer.fit(train_dataloader=dummy_dataloader, log_interval=log_interval, epochs=10)
        assert any(
            rec.levelname == "ERROR" and "log_interval cannot be less than 1" in rec.message for rec in caplog.records
        )

    def test_fit_no_epochs_left(self, dummy_trainer, dummy_dataloader, monkeypatch, caplog):
        trainer = dummy_trainer
        dummy_trainer.current_epoch = 10
        with pytest.raises(ValueError, match="No epochs left to train"):
            trainer.fit(train_dataloader=dummy_dataloader, epochs=10)
        assert any(rec.levelname == "ERROR" and "No epochs left to train" in rec.message for rec in caplog.records)

    def test_fit_with_invalid_mode(self, dummy_trainer, dummy_dataloader, monkeypatch, caplog):
        trainer = dummy_trainer
        with pytest.raises(ValueError, match="Unknown mode"):
            trainer.fit(train_dataloader=dummy_dataloader, epochs=10, mode="test")
        assert any(rec.levelname == "ERROR" and "Unknown mode" in rec.message for rec in caplog.records)

    def test_fit_without_train_dataloader(self, dummy_trainer, monkeypatch, caplog):
        trainer = dummy_trainer
        with pytest.raises(ValueError, match="train_dataloader cannot be none"):
            trainer.fit(train_dataloader=None, epochs=10)
        assert any(
            rec.levelname == "ERROR" and "train_dataloader cannot be none" in rec.message for rec in caplog.records
        )

    def test_fit_with_empty_train_dataloader(self, dummy_trainer, dummy_empty_dataloader, monkeypatch, caplog):
        trainer = dummy_trainer
        with pytest.raises(ValueError, match="train_dataloader dataset cannot be empty"):
            trainer.fit(train_dataloader=dummy_empty_dataloader, epochs=10)
        assert any(
            rec.levelname == "ERROR" and "train_dataloader dataset cannot be empty" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("dataloader", [10, 0.5, "test"])
    def test_fit_with_unsupported_train_dataloader_type(self, dummy_trainer, dataloader, monkeypatch, caplog):
        trainer = dummy_trainer
        with pytest.raises(ValueError, match="Unsupported train_dataloader type"):
            trainer.fit(train_dataloader=dataloader, epochs=10)
        assert any(
            rec.levelname == "ERROR" and "Unsupported train_dataloader type" in rec.message for rec in caplog.records
        )

    def test_fit_with_empty_val_dataloader(
        self, dummy_trainer, dummy_dataloader, dummy_empty_dataloader, monkeypatch, caplog
    ):
        trainer = dummy_trainer
        with pytest.raises(ValueError, match="val_dataloader dataset cannot be empty"):
            trainer.fit(train_dataloader=dummy_dataloader, val_dataloader=dummy_empty_dataloader, epochs=10)
        assert any(
            rec.levelname == "ERROR" and "val_dataloader dataset cannot be empty" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("dataloader", [10, 0.5, "test"])
    def test_fit_with_unsupported_val_dataloader_type(
        self, dummy_trainer, dummy_dataloader, dataloader, monkeypatch, caplog
    ):
        trainer = dummy_trainer
        with pytest.raises(ValueError, match="Unsupported val_dataloader type"):
            trainer.fit(train_dataloader=dummy_dataloader, val_dataloader=dataloader, epochs=10)
        assert any(
            rec.levelname == "ERROR" and "Unsupported val_dataloader type" in rec.message for rec in caplog.records
        )

    def test_fit_with_without_val_dataloader_with_val_save_criteria(
        self, dummy_trainer, dummy_dataloader, monkeypatch, caplog
    ):
        trainer = dummy_trainer
        with pytest.raises(ValueError, match="With save criterion val/loss, val_dataloader cannot be None"):
            trainer.fit(train_dataloader=dummy_dataloader, val_dataloader=None, epochs=10, save_criterion="val/loss")
        assert any(
            rec.levelname == "ERROR" and "With save criterion val/loss, val_dataloader cannot be None" in rec.message
            for rec in caplog.records
        )

    def test_save_checkpoint(
        self, dummy_model, dummy_config, dummy_loss_function, dummy_optimizer, dummy_metrics, tmp_path, dummy_logger
    ):
        log_dir = tmp_path / "test_dir"
        trainer = Trainer(
            model=dummy_model,
            config=dummy_config,
            loss_function=dummy_loss_function,
            optimizer=dummy_optimizer,
            metrics=dummy_metrics,
            log_dir=log_dir,
            device="cpu",
            logger=dummy_logger,
        )
        trainer.current_epoch = 5
        trainer.best_value = 0.5
        trainer.save_criterion = "val/loss"
        trainer.save_checkpoint(is_best=True)

        best_path = tmp_path / "test_dir" / f"{trainer.model_name}_best.pt"
        assert best_path.exists()
        checkpoint = torch.load(best_path)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "scheduler_state_dict" in checkpoint
        assert "metrics_history" in checkpoint
        assert "config" in checkpoint
        assert checkpoint["epoch"] == 5
        assert checkpoint["model_name"] == "TestModel"
        assert checkpoint["save_criterion"] == "val/loss"
        assert checkpoint["best_value"] == 0.5

    def test_save_checkpoint_intermediate(self, dummy_trainer, tmp_path):
        trainer = dummy_trainer
        trainer.current_epoch = 2
        trainer.save_checkpoint(is_best=False)
        checkpoints_dir = tmp_path / "test_dir" / "checkpoints"
        files = list(checkpoints_dir.glob(f"{trainer.model_name}_epoch_2_*.pt"))
        assert len(files) == 1

    def test_load_checkpoint(self, dummy_trainer, dummy_combo_loss_function, dummy_loss_function, tmp_path, caplog):
        trainer = dummy_trainer
        trainer.current_epoch = 10
        trainer.best_value = 0.75
        trainer.save_criterion = "val/iou"
        trainer.metrics_history = {"train": {"loss": [0.1]}, "val": {"loss": [0.2]}}
        trainer.save_checkpoint(is_best=True)

        trainer.current_epoch = 5
        trainer.best_value = 0.5
        trainer.save_criterion = "train/loss"
        trainer.metrics_history = {"train": {"loss": [0.9]}, "val": {"loss": [0.8]}}

        best_path = tmp_path / "test_dir" / f"{trainer.model_name}_best.pt"
        trainer.load_checkpoint(best_path, load_optimizer=True, load_scheduler=False)

        assert trainer.current_epoch == 10
        assert trainer.best_value == 0.75
        assert trainer.save_criterion == "val/iou"
        assert trainer.metrics_history == {"train": {"loss": [0.1]}, "val": {"loss": [0.2]}}
        for p_loaded, p_orig in zip(trainer.model.parameters(), dummy_trainer.model.parameters()):
            assert torch.equal(p_loaded, p_orig)

        assert any(rec.levelname == "INFO" and "Checkpoint loaded from" in rec.message for rec in caplog.records)

    def test_load_checkpoint_with_another_config(self, dummy_trainer, tmp_path, caplog):
        trainer = dummy_trainer
        trainer.save_checkpoint(is_best=True)
        trainer.config = {}
        best_path = tmp_path / "test_dir" / f"{trainer.model_name}_best.pt"
        trainer.load_checkpoint(best_path, load_optimizer=True, load_scheduler=False)

        assert any(rec.levelname == "INFO" and "Checkpoint loaded from" in rec.message for rec in caplog.records)
        assert any(
            rec.levelname == "WARNING" and "Configuration mismatch detected" in rec.message for rec in caplog.records
        )

    def test_load_trainer(
        self, dummy_trainer, dummy_logger, dummy_loss_function, dummy_metrics, tmp_path, caplog, monkeypatch
    ):
        trainer = dummy_trainer
        trainer.current_epoch = 10
        trainer.best_value = 0.75
        trainer.save_criterion = "val/iou"
        trainer.metrics_history = {"train": {"loss": [0.1]}, "val": {"loss": [0.2]}}
        trainer.save_checkpoint(is_best=True)

        monkeypatch.setattr("src.utils.factories.model_factory.create_model", lambda cfg: trainer.model)
        monkeypatch.setattr("src.utils.factories.loss_fn_factory.create_loss", lambda cfg: trainer.loss_function)
        monkeypatch.setattr("src.utils.factories.metrics_factory.create_metrics", lambda cfg: trainer.metrics)
        monkeypatch.setattr(
            "src.utils.factories.optimizer_factory.create_optimizer", lambda cfg, model: trainer.optimizer
        )
        monkeypatch.setattr(
            "src.utils.factories.scheduler_factory.create_scheduler", lambda cfg, optimizer: trainer.scheduler
        )

        trainer.current_epoch = 5
        trainer.best_value = 0.5
        trainer.save_criterion = "train/loss"
        trainer.metrics_history = {"train": {"loss": [0.9]}, "val": {"loss": [0.8]}}

        log_dir = tmp_path / "test_dir"
        best_path = log_dir / f"{trainer.model_name}_best.pt"
        trainer = Trainer.load_trainer(best_path, log_dir, logger=dummy_logger)

        assert trainer.current_epoch == 10
        assert trainer.best_value == 0.75
        assert trainer.save_criterion == "val/iou"
        assert trainer.metrics_history == {"train": {"loss": [0.1]}, "val": {"loss": [0.2]}}
        assert trainer.loss_function == dummy_loss_function
        assert trainer.metrics == dummy_metrics
        for p_loaded, p_orig in zip(trainer.model.parameters(), dummy_trainer.model.parameters()):
            assert torch.equal(p_loaded, p_orig)

    @pytest.mark.parametrize("config", [None, {}])
    def test_load_trainer_without_config_in_checkpoint(self, dummy_trainer, config, dummy_logger, tmp_path, caplog):
        trainer = dummy_trainer
        trainer.config = config
        trainer.save_checkpoint(is_best=True)

        log_dir = tmp_path / "test_dir"
        best_path = log_dir / f"{trainer.model_name}_best.pt"

        with pytest.raises(ValueError, match="Checkpoint does not contain config. Cannot restore components."):
            Trainer.load_trainer(best_path, log_dir, logger=dummy_logger)
