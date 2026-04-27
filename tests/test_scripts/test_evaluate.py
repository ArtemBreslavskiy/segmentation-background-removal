from unittest.mock import ANY, MagicMock, patch

import pytest

from scripts.evaluate import evaluate


class TestEvaluate:
    @patch("scripts.evaluate.get_test_dataloader")
    @patch("scripts.evaluate.create_metrics")
    @patch("scripts.evaluate.Tester")
    @patch("scripts.evaluate.ProjectPaths")
    def test_evaluate_success(
        self,
        mock_project_paths_class,
        mock_tester_class,
        mock_create_metrics,
        mock_get_test_loader,
        mock_paths_with_evaluate,
        full_config,
        mock_logger,
        trained_checkpoint,
    ):
        mock_paths_with_evaluate.SAVED_CHECKPOINTS = mock_paths_with_evaluate.base / "checkpoints"
        checkpoint_path = mock_paths_with_evaluate.SAVED_CHECKPOINTS / "test_model_best.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(trained_checkpoint.read_bytes())
        mock_project_paths_class.return_value = mock_paths_with_evaluate

        logger = mock_logger

        mock_tester = MagicMock()
        mock_tester.evaluate.return_value = {"accuracy": 0.95, "iou": 0.85}
        mock_tester_class.load_tester.return_value = mock_tester

        mock_metrics = {"accuracy": MagicMock(), "iou": MagicMock()}
        mock_create_metrics.return_value = mock_metrics

        mock_dataloader = MagicMock()
        mock_get_test_loader.return_value = mock_dataloader

        evaluate(logger)

        mock_tester_class.load_tester.assert_called_once_with(
            path=checkpoint_path,
            log_dir=mock_paths_with_evaluate.SAVED_BEST_MODEL_TESTS,
            device="cpu",
        )
        mock_get_test_loader.assert_called_once_with(full_config, mock_paths_with_evaluate)
        mock_create_metrics.assert_called_once_with(full_config)
        mock_tester.evaluate.assert_called_once_with(dataloader=mock_dataloader, metrics=mock_metrics)

        logger.info.assert_any_call("EVALUATION STARTED")
        logger.info.assert_any_call("TEST METRICS RESULTS")
        logger.info.assert_any_call("%s: %s", "accuracy", 0.95)
        logger.info.assert_any_call("%s: %s", "iou", 0.85)
        logger.info.assert_any_call("EVALUATION COMPLETED SUCCESSFULLY")

    @patch("scripts.evaluate.ProjectPaths")
    def test_evaluate_checkpoint_not_found(
        self,
        mock_project_paths_class,
        mock_paths_with_evaluate,
        full_config,
        mock_logger,
    ):
        mock_paths_with_evaluate.SAVED_CHECKPOINTS = mock_paths_with_evaluate.base / "checkpoints"
        mock_project_paths_class.return_value = mock_paths_with_evaluate
        logger = mock_logger

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            evaluate(logger)

        logger.critical.assert_called_once_with("Evaluation failed. Check error log for details.")
        logger.exception.assert_called_once()
        call_args = logger.exception.call_args[0][0]
        assert "Checkpoint not found at:" in call_args

    @patch("scripts.evaluate.get_test_dataloader")
    @patch("scripts.evaluate.create_metrics")
    @patch("scripts.evaluate.Tester")
    @patch("scripts.evaluate.ProjectPaths")
    def test_evaluate_tester_loading_failure(
        self,
        mock_project_paths_class,
        mock_tester_class,
        mock_create_metrics,
        mock_get_test_loader,
        mock_paths_with_evaluate,
        full_config,
        mock_logger,
        trained_checkpoint,
    ):
        mock_paths_with_evaluate.SAVED_CHECKPOINTS = mock_paths_with_evaluate.base / "checkpoints"
        checkpoint_path = mock_paths_with_evaluate.SAVED_CHECKPOINTS / "test_model_best.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(trained_checkpoint.read_bytes())
        mock_project_paths_class.return_value = mock_paths_with_evaluate

        logger = mock_logger

        mock_tester_class.load_tester.side_effect = Exception("Load failed")

        with pytest.raises(Exception, match="Load failed"):
            evaluate(logger)

        logger.critical.assert_called_once_with("Evaluating failed. Check error log for details.")
        logger.exception.assert_called_once_with("Tester loading failed: %s", ANY)

    @patch("scripts.evaluate.get_test_dataloader")
    @patch("scripts.evaluate.create_metrics")
    @patch("scripts.evaluate.Tester")
    @patch("scripts.evaluate.ProjectPaths")
    def test_evaluate_evaluation_error(
        self,
        mock_project_paths_class,
        mock_tester_class,
        mock_create_metrics,
        mock_get_test_loader,
        mock_paths_with_evaluate,
        full_config,
        mock_logger,
        trained_checkpoint,
    ):
        mock_paths_with_evaluate.SAVED_CHECKPOINTS = mock_paths_with_evaluate.base / "checkpoints"
        checkpoint_path = mock_paths_with_evaluate.SAVED_CHECKPOINTS / "test_model_best.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(trained_checkpoint.read_bytes())
        mock_project_paths_class.return_value = mock_paths_with_evaluate

        logger = mock_logger

        mock_tester = MagicMock()
        mock_tester.evaluate.side_effect = Exception("Evaluation error")
        mock_tester_class.load_tester.return_value = mock_tester

        mock_get_test_loader.return_value = MagicMock()
        mock_create_metrics.return_value = {}

        with pytest.raises(Exception, match="Evaluation error"):
            evaluate(logger)

        logger.error.assert_called_once_with("Evaluation failed. Check error log for details.")
        logger.exception.assert_called_once_with("Error during evaluation")
