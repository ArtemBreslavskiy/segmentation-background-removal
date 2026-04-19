import copy

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.engine.Trainer import Trainer
from src.losses.ComboLoss import ComboLoss


class TestTrainer:

    @pytest.mark.parametrize("use_scheduler", [True, False])
    def test_init(
        self, model, config, loss, optimizer, metrics, log_dir, device, use_scheduler
    ):
        scheduler = (
            optim.lr_scheduler.StepLR(optimizer, step_size=1) if use_scheduler else None
        )
        trainer = Trainer(
            model=model,
            config=config,
            loss_function=loss,
            optimizer=optimizer,
            metrics=metrics,
            scheduler=scheduler,
            log_dir=log_dir,
            device=device,
        )
        assert trainer.scheduler == scheduler
        assert isinstance(trainer.metrics_history, dict)

    def test_train_epoch(self, trainer, train_loader):
        result = trainer.train_epoch(train_loader)
        assert "loss" in result
        assert len(trainer.metrics_history["train"]["loss"]) > 0

    def test_validate_epoch(self, trainer, val_loader):
        result = trainer.validate_epoch(val_loader)
        assert "loss" in result
        assert len(trainer.metrics_history["val"]["loss"]) > 0

    @pytest.mark.parametrize("mode", ["min", "max"])
    def test_fit_runs(self, trainer, train_loader, val_loader, mode):
        trainer.fit(train_loader, val_loader, epochs=2, mode=mode)

    def test_fit_train_only(self, trainer, train_loader):
        trainer.fit(
            train_loader, val_dataloader=None, epochs=1, save_criterion="train/loss"
        )

    def test_fit_invalid_mode(self, trainer, train_loader):
        with pytest.raises(ValueError):
            trainer.fit(train_loader, epochs=1, mode="wrong")

    def test_fit_empty_train_loader(self, trainer):
        with pytest.raises(ValueError):
            trainer.fit([], epochs=1)

    def test_fit_empty_val_loader(self, trainer, train_loader):
        with pytest.raises(ValueError):
            trainer.fit(train_loader, val_dataloader=[], epochs=1)

    def test_fit_invalid_criterion(self, trainer, train_loader):
        with pytest.raises(ValueError):
            trainer.fit(train_loader, epochs=1, save_criterion="invalid")

    def test_early_stopping(self, trainer, train_loader, val_loader):
        trainer.fit(
            train_loader,
            val_loader,
            epochs=5,
            early_stopping_patience=1,
        )

    def test_scheduler_step(self, trainer, train_loader, val_loader):
        trainer.scheduler = optim.lr_scheduler.StepLR(trainer.optimizer, step_size=1)
        trainer.fit(train_loader, val_loader, epochs=2)

    def test_reduce_on_plateau_scheduler(self, trainer, train_loader, val_loader):
        trainer.scheduler = optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer)
        trainer.fit(train_loader, val_loader, epochs=2)

    def test_save_checkpoint(self, trainer, train_loader):
        trainer.train_epoch(train_loader)
        trainer.save_checkpoint(is_best=False)
        files = list((trainer.log_dir / "checkpoints").glob("*.pt"))
        assert len(files) > 0

    def test_save_best_checkpoint(self, trainer, train_loader):
        trainer.train_epoch(train_loader)
        trainer.save_checkpoint(is_best=True)
        best_file = trainer.log_dir / f"{trainer.model_name}_best.pt"
        assert best_file.exists()

    def test_load_checkpoint(self, trainer, train_loader):
        trainer.train_epoch(train_loader)
        trainer.save_checkpoint(is_best=True)
        path = trainer.log_dir / f"{trainer.model_name}_best.pt"
        trainer.load_checkpoint(path)

    def test_load_checkpoint_without_optimizer(self, trainer, train_loader):
        trainer.train_epoch(train_loader)
        trainer.save_checkpoint(is_best=True)
        path = trainer.log_dir / f"{trainer.model_name}_best.pt"
        trainer.load_checkpoint(path, load_optimizer=False)

    def test_load_checkpoint_without_scheduler(self, trainer, train_loader):
        trainer.scheduler = optim.lr_scheduler.StepLR(trainer.optimizer, step_size=1)
        trainer.train_epoch(train_loader)
        trainer.save_checkpoint(is_best=True)
        path = trainer.log_dir / f"{trainer.model_name}_best.pt"
        trainer.load_checkpoint(path, load_scheduler=False)

    def test_checkpoint_restores_state(self, trainer, train_loader):
        trainer.train_epoch(train_loader)
        trainer.save_checkpoint(is_best=True)
        path = trainer.log_dir / f"{trainer.model_name}_best.pt"

        before_epoch = trainer.current_epoch
        trainer.current_epoch = 0
        trainer.load_checkpoint(path)

        assert trainer.current_epoch == before_epoch

    def test_fit_saves_best(self, trainer, train_loader, val_loader):
        trainer.fit(train_loader, val_loader, epochs=2)
        best_file = trainer.log_dir / f"{trainer.model_name}_best.pt"
        assert best_file.exists()

    def test_log_interval_checkpoint(self, trainer, train_loader, val_loader):
        trainer.fit(train_loader, val_loader, epochs=2, log_interval=1)
        files = list(trainer.log_dir.rglob("*.pt"))
        assert len(files) > 0

    def test_load_trainer(self, trainer, train_loader, monkeypatch):
        trainer.train_epoch(train_loader)
        trainer.save_checkpoint(is_best=True)
        path = trainer.log_dir / f"{trainer.model_name}_best.pt"

        def fake_model(config):
            return copy.deepcopy(trainer.model)

        def fake_loss(config):
            return nn.BCEWithLogitsLoss()

        def fake_metrics(config):
            return {}

        def fake_optimizer(config, model):
            return optim.SGD(model.parameters(), lr=0.1)

        def fake_scheduler(config, optimizer):
            return None

        monkeypatch.setattr("src.utils.factory.create_model", fake_model)
        monkeypatch.setattr("src.utils.factory.create_loss", fake_loss)
        monkeypatch.setattr("src.utils.factory.create_metrics", fake_metrics)
        monkeypatch.setattr("src.utils.factory.create_optimizer", fake_optimizer)
        monkeypatch.setattr("src.utils.factory.create_scheduler", fake_scheduler)

        loaded = Trainer.load_trainer(
            path, log_dir=trainer.log_dir, device=trainer.device
        )

        assert isinstance(loaded, Trainer)
        assert loaded.current_epoch == trainer.current_epoch
        assert loaded.model_name == trainer.model_name

    @pytest.mark.parametrize("criterion", ["train/loss", "val/loss"])
    def test_fit_save_criterion(self, trainer, train_loader, val_loader, criterion):
        if criterion.startswith("val"):
            trainer.fit(train_loader, val_loader, epochs=2, save_criterion=criterion)
        else:
            trainer.fit(train_loader, epochs=2, save_criterion=criterion)

    def test_metrics_history_updated(self, trainer, train_loader, val_loader):
        trainer.fit(train_loader, val_loader, epochs=2)
        assert len(trainer.metrics_history["train"]["loss"]) > 0
        assert len(trainer.metrics_history["val"]["loss"]) > 0

    @pytest.mark.parametrize("weights", [None, [0.5, 0.5], [2.0, 1.0]])
    def test_combo_loss_in_training(
        self, model, config, optimizer, metrics, log_dir, device, train_loader, weights
    ):
        loss = ComboLoss([nn.BCEWithLogitsLoss(), nn.L1Loss()], weights=weights)
        trainer = Trainer(
            model=model,
            config=config,
            loss_function=loss,
            optimizer=optimizer,
            metrics=metrics,
            log_dir=log_dir,
            device=device,
        )
        result = trainer.train_epoch(train_loader)
        assert "loss" in result
        assert isinstance(result["loss"], float)

    def test_combo_loss_components_logged(
        self, model, config, optimizer, metrics, log_dir, device, train_loader
    ):
        loss = ComboLoss([nn.BCEWithLogitsLoss(), nn.L1Loss()])
        trainer = Trainer(
            model=model,
            config=config,
            loss_function=loss,
            optimizer=optimizer,
            metrics=metrics,
            log_dir=log_dir,
            device=device,
        )
        trainer.train_epoch(train_loader)
        history = trainer.metrics_history["train"]
        assert "loss" in history

    def test_combo_loss_forward_matches_manual(self):
        logits = torch.randn(4, 1, 16, 16)
        targets = torch.randn(4, 1, 16, 16)

        l1 = nn.L1Loss()
        l2 = nn.MSELoss()

        combo = ComboLoss([l1, l2], weights=[1.0, 1.0])
        value = combo(logits, targets)

        expected = (l1(logits, targets) + l2(logits, targets)) / 2
        assert torch.allclose(value, expected)

    def test_combo_loss_components(self):
        logits = torch.randn(2, 1, 8, 8)
        targets = torch.randn(2, 1, 8, 8)

        combo = ComboLoss([nn.L1Loss(), nn.MSELoss()])
        total, components = combo.forward_with_components(logits, targets)

        assert isinstance(components, dict)
        assert len(components) == 2
        assert isinstance(total, torch.Tensor)

    @pytest.mark.parametrize("weights", [[1.0, 3.0], [0.2, 0.8]])
    def test_combo_loss_weighting(self, weights):
        logits = torch.randn(2, 1, 8, 8)
        targets = torch.randn(2, 1, 8, 8)

        l1 = nn.L1Loss()
        l2 = nn.MSELoss()

        combo = ComboLoss([l1, l2], weights=weights)
        total, raw = combo.forward_with_components(logits, targets)

        norm = [w / sum(weights) for w in weights]
        expected = raw[combo.names[0]] * norm[0] + raw[combo.names[1]] * norm[1]

        assert torch.allclose(total, expected)

    def test_combo_loss_names_unique(self):
        combo = ComboLoss([nn.L1Loss(), nn.L1Loss(), nn.L1Loss()])
        assert len(set(combo.names)) == 3

    def test_combo_loss_count(self):
        combo = ComboLoss([nn.L1Loss(), nn.MSELoss()])
        assert combo.count == 2

    def test_combo_loss_invalid_empty(self):
        with pytest.raises(ValueError):
            ComboLoss([])

    def test_combo_loss_invalid_weights(self):
        with pytest.raises(ValueError):
            ComboLoss([nn.L1Loss(), nn.MSELoss()], weights=[1.0])

    def test_combo_loss_zero_weights(self):
        with pytest.raises(ValueError):
            ComboLoss([nn.L1Loss(), nn.MSELoss()], weights=[0.0, 0.0])

    def test_combo_loss_with_fit(
        self,
        model,
        config,
        optimizer,
        metrics,
        log_dir,
        device,
        train_loader,
        val_loader,
    ):
        loss = ComboLoss([nn.BCEWithLogitsLoss(), nn.L1Loss()])
        trainer = Trainer(
            model=model,
            config=config,
            loss_function=loss,
            optimizer=optimizer,
            metrics=metrics,
            log_dir=log_dir,
            device=device,
        )
        trainer.fit(train_loader, val_loader, epochs=2)
        assert len(trainer.metrics_history["train"]["loss"]) > 0
