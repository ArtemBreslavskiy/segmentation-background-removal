import pytest
import torch

from src.engine.Tester import Tester


class TestTester:
    def test_evaluate_with_empty_metrics(self, dummy_tester, dummy_dataloader, caplog):
        tester = dummy_tester
        with pytest.raises(ValueError, match="metrics cannot be none or empty"):
            tester.evaluate(dummy_dataloader, {})
        assert any(
            rec.levelname == "ERROR" and "metrics cannot be none or empty" in rec.message for rec in caplog.records
        )

    @pytest.mark.parametrize("metrics", [10, 0.5, "test"])
    def test_evaluate_with_unsupported_metrics_format(self, dummy_tester, dummy_dataloader, metrics, caplog):
        tester = dummy_tester
        with pytest.raises(ValueError, match="Unsupported metrics format"):
            tester.evaluate(dummy_dataloader, metrics)
        assert any(rec.levelname == "ERROR" and "Unsupported metrics format" in rec.message for rec in caplog.records)

    def test_load_tester(self, dummy_trainer, dummy_logger, tmp_path, caplog, monkeypatch):
        trainer = dummy_trainer
        trainer.save_checkpoint(is_best=True)

        monkeypatch.setattr("src.utils.factories.model_factory.create_model", lambda cfg: trainer.model)
        monkeypatch.setattr("src.utils.factories.loss_fn_factory.create_loss", lambda cfg: trainer.loss_function)
        monkeypatch.setattr("src.utils.factories.metrics_factory.create_metrics", lambda cfg: trainer.metrics)

        log_dir = tmp_path / "test_dir"
        best_path = log_dir / f"{trainer.model_name}_best.pt"
        tester = Tester.load_tester(best_path, log_dir, logger=dummy_logger)

        assert trainer.loss_function == tester.loss_function
        assert trainer.metrics == tester.metrics
        for p_loaded, p_orig in zip(trainer.model.parameters(), dummy_trainer.model.parameters()):
            assert torch.equal(p_loaded, p_orig)

    @pytest.mark.parametrize("config", [None, {}])
    def test_load_tester_without_config_in_checkpoint(self, dummy_trainer, config, dummy_logger, tmp_path, caplog):
        trainer = dummy_trainer
        trainer.config = config
        trainer.save_checkpoint(is_best=True)

        log_dir = tmp_path / "test_dir"
        best_path = log_dir / f"{trainer.model_name}_best.pt"

        with pytest.raises(ValueError, match="Checkpoint does not contain config. Cannot restore components."):
            Tester.load_tester(best_path, log_dir, logger=dummy_logger)
