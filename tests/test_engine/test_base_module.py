import pytest
import torch
import torch.utils.data as data

from src.engine.BaseModule import BaseModule


class TestBaseModule:

    @pytest.mark.parametrize("device", ["cpu"])
    def test_init_sets_attributes(self, model, config, loss, optimizer, metrics, log_dir, device):
        module = BaseModule(
            model=model,
            config=config,
            loss_function=loss,
            optimizer=optimizer,
            metrics=metrics,
            log_dir=log_dir,
            device=device,
        )
        assert module.model is not None
        assert module.optimizer is optimizer
        assert module.loss_function is loss
        assert module.device.type == device
        assert module.model_name == config["model"]["model_name"]
        assert isinstance(module.metrics, dict)

    @pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
    def test_device_validate_cpu(self, model, config, loss, device):
        module = BaseModule(model=model, config=config, loss_function=loss, device=device)
        assert module.device.type == "cpu"

    @pytest.mark.parametrize("device", ["invalid", "gpu:99"])
    def test_device_invalid(self, model, config, loss, device):
        with pytest.raises(ValueError):
            BaseModule(model=model, config=config, loss_function=loss, device=device)

    def test_ensure_log_dir_created(self, model, config, loss, tmp_path):
        module = BaseModule(
            model=model,
            config=config,
            loss_function=loss,
            log_dir=tmp_path,
        )
        assert module.log_dir.exists()
        assert (module.log_dir / "checkpoints").exists()

    @pytest.mark.parametrize("batch_type", ["tuple", "dict"])
    def test_move_batch(self, trainer, batch_type):
        x = torch.randn(2, 1, 32, 32)
        y = torch.randn(2, 1, 32, 32)

        if batch_type == "tuple":
            batch = (x, y)
        else:
            batch = {"image": x, "mask": y}

        moved = trainer._move_batch_to_device(batch)

        if batch_type == "tuple":
            assert moved[0].device == trainer.device
            assert moved[1].device == trainer.device
        else:
            assert moved["image"].device == trainer.device
            assert moved["mask"].device == trainer.device

    @pytest.mark.parametrize("batch_type", ["tuple", "dict"])
    def test_unpack_batch(self, trainer, batch_type):
        x = torch.randn(2, 1, 32, 32)
        y = torch.randn(2, 1, 32, 32)

        if batch_type == "tuple":
            bx, by = trainer._unpack_batch((x, y))
        else:
            bx, by = trainer._unpack_batch({"image": x, "mask": y})

        assert torch.equal(bx, x)
        assert torch.equal(by, y)

    def test_unpack_invalid(self, trainer):
        with pytest.raises(ValueError):
            trainer._unpack_batch((torch.randn(1),))

    @pytest.mark.parametrize("return_components", [True, False])
    def test_compute_loss(self, trainer, return_components):
        preds = torch.randn(2, 1, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32)).float()

        result = trainer._compute_loss(preds, targets, return_components=return_components)

        if return_components:
            loss, comp = result
            assert isinstance(loss, torch.Tensor)
            assert isinstance(comp, dict)
        else:
            assert isinstance(result, torch.Tensor)

    def test_update_metrics(self, trainer):
        preds = torch.randn(2, 1, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32))
        trainer._update_metrics(preds, targets, trainer.metrics)
        values = trainer._compute_metrics(trainer.metrics)
        assert isinstance(values, dict)

    def test_reset_metrics(self, trainer):
        trainer._reset_metrics(trainer.metrics)
        for metric in trainer.metrics.values():
            assert metric is not None

    @pytest.mark.parametrize("mode", ["train", "val"])
    def test_run_epoch_modes(self, trainer, train_loader, val_loader, mode):
        loader = train_loader if mode == "train" else val_loader
        result = trainer.run_epoch(loader, mode=mode)
        assert "loss" in result

    def test_run_epoch_test(self, tester, test_loader, metrics):
        result = tester.run_epoch(test_loader, mode="test", metrics=metrics)
        assert "loss" in result

    @pytest.mark.parametrize("mode", ["wrong", "invalid", "TRAINN"])
    def test_run_epoch_invalid_mode(self, trainer, train_loader, mode):
        with pytest.raises(ValueError):
            trainer.run_epoch(train_loader, mode=mode)

    def test_run_epoch_no_optimizer(self, model, config, loss, train_loader):
        module = BaseModule(model=model, config=config, loss_function=loss)
        with pytest.raises(RuntimeError):
            module.run_epoch(train_loader, mode="train")

    def test_run_epoch_empty_loader(self, trainer):
        loader = data.DataLoader([], batch_size=2)
        with pytest.raises(ValueError):
            trainer.run_epoch(loader, mode="train")

    def test_run_epoch_none_loader(self, trainer):
        with pytest.raises(ValueError):
            trainer.run_epoch(None, mode="train")

    def test_metrics_computation_after_epoch(self, trainer, train_loader):
        result = trainer.run_epoch(train_loader, mode="train")
        assert isinstance(result["loss"], float)

    def test_epoch_updates_metrics(self, trainer, train_loader):
        trainer.run_epoch(train_loader, mode="train")
        values = trainer._compute_metrics(trainer.metrics)
        assert isinstance(values, dict)

    def test_loss_backward_updates_weights(self, trainer, train_loader):
        before = [p.clone().detach() for p in trainer.model.parameters() if p.requires_grad]
        trainer.run_epoch(train_loader, mode="train")
        after = [p for p in trainer.model.parameters() if p.requires_grad]
        assert any(not torch.equal(b, a) for b, a in zip(before, after))

    def test_eval_mode_no_grad(self, trainer, val_loader):
        before = [p.clone().detach() for p in trainer.model.parameters() if p.requires_grad]
        trainer.run_epoch(val_loader, mode="val")
        after = [p for p in trainer.model.parameters() if p.requires_grad]
        assert all(torch.equal(b, a) for b, a in zip(before, after))
