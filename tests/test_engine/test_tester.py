import pytest
import torch

from src.engine.Tester import Tester


class TestTester:

    @pytest.mark.parametrize("metrics_used", [True, False])
    def test_evaluate(self, tester, test_loader, metrics, metrics_used):
        m = metrics if metrics_used else {}
        result = tester.evaluate(test_loader, m)
        assert "loss" in result

    @pytest.mark.parametrize("repeat", [1, 2, 3])
    def test_evaluate_multiple_runs(self, tester, test_loader, repeat):
        for _ in range(repeat):
            result = tester.evaluate(test_loader, tester.metrics)
            assert isinstance(result, dict)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_evaluate_different_loader_sizes(self, tester, dataset, batch_size):
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        result = tester.evaluate(loader, tester.metrics)
        assert "loss" in result

    @pytest.mark.parametrize("metrics_count", [0, 1, 2])
    def test_metrics_variants(self, tester, test_loader, metrics, metrics_count):
        selected = dict(list(metrics.items())[:metrics_count])
        result = tester.evaluate(test_loader, selected)
        assert "loss" in result

    @pytest.mark.parametrize("runs", [1, 2])
    def test_files_saved(self, tester, test_loader, runs):
        for _ in range(runs):
            tester.evaluate(test_loader, tester.metrics)
        assert any(tester.log_dir.glob("*.pt"))

    def test_empty_loader(self, tester):
        with pytest.raises(ValueError):
            tester.evaluate([], tester.metrics)

    @pytest.mark.parametrize("device_transfer", [True])
    def test_metrics_device(self, tester, test_loader, device_transfer):
        tester.evaluate(test_loader, tester.metrics)
        for metric in tester.metrics.values():
            assert hasattr(metric, "to")

    @pytest.mark.parametrize("check_float", [True])
    def test_values_are_float(self, tester, test_loader, check_float):
        result = tester.evaluate(test_loader, tester.metrics)
        for v in result.values():
            assert isinstance(v, float)

    @pytest.mark.parametrize("epoch", [0, 3, 7])
    def test_epoch_not_modified(self, tester, test_loader, epoch):
        tester.current_epoch = epoch
        tester.evaluate(test_loader, tester.metrics)
        assert tester.current_epoch == epoch

    @pytest.mark.parametrize("calls", [1, 2])
    def test_save_metrics_called(self, tester, test_loader, monkeypatch, calls):
        counter = {"c": 0}

        def fake_save(_):
            counter["c"] += 1

        monkeypatch.setattr(tester, "_save_metrics", fake_save)

        for _ in range(calls):
            tester.evaluate(test_loader, tester.metrics)

        assert counter["c"] == calls

    @pytest.mark.parametrize("metrics_used", [True, False])
    def test_return_keys(self, tester, test_loader, metrics, metrics_used):
        m = metrics if metrics_used else {}
        result = tester.evaluate(test_loader, m)
        assert "loss" in result

    @pytest.mark.parametrize("runs", [1, 2])
    def test_log_files_exist(self, tester, test_loader, runs):
        for _ in range(runs):
            tester.evaluate(test_loader, tester.metrics)
        assert any(tester.log_dir.glob("*.pt"))

    def test_init(self, model, config, loss, metrics, log_dir, device):
        tester = Tester(
            model=model,
            config=config,
            loss_function=loss,
            metrics=metrics,
            log_dir=log_dir,
            device=device,
            model_name="test_model",
        )
        assert isinstance(tester, Tester)

    @pytest.mark.parametrize("metrics_used", [True, False])
    def test_loss_present(self, tester, test_loader, metrics, metrics_used):
        m = metrics if metrics_used else {}
        result = tester.evaluate(test_loader, m)
        assert "loss" in result
