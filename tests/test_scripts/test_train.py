import ctypes
import platform
from unittest.mock import ANY, MagicMock, patch

import pytest

from scripts.train import train


class TestTrain:
    @patch("scripts.train.get_train_dataloader")
    @patch("scripts.train.get_val_dataloader")
    @patch("scripts.train.ProjectPaths")
    def test_train_success(
        self,
        mock_project_paths_class,
        mock_get_val_loader,
        mock_get_train_loader,
        mock_paths,
        full_config,
        mock_logger,
    ):
        mock_project_paths_class.return_value = mock_paths
        logger = mock_logger

        mock_train_loader = MagicMock()
        mock_train_loader.dataset = MagicMock()
        mock_train_loader.dataset.__len__.return_value = 100
        mock_get_train_loader.return_value = mock_train_loader

        mock_val_loader = MagicMock()
        mock_val_loader.dataset = MagicMock()
        mock_val_loader.dataset.__len__.return_value = 50
        mock_get_val_loader.return_value = mock_val_loader

        with patch("scripts.train.create_model", return_value=MagicMock()), patch(
            "scripts.train.create_optimizer", return_value=MagicMock()
        ), patch("scripts.train.create_loss", return_value=MagicMock()), patch(
            "scripts.train.create_metrics", return_value={}
        ), patch(
            "scripts.train.create_scheduler", return_value=MagicMock()
        ), patch(
            "scripts.train.Trainer"
        ) as mock_trainer_class:
            mock_trainer = MagicMock()
            mock_trainer_class.return_value = mock_trainer

            train(logger)

        mock_trainer_class.assert_called_once()
        mock_trainer.fit.assert_called_once_with(
            train_dataloader=mock_train_loader,
            val_dataloader=mock_val_loader,
            epochs=full_config["learning"]["epochs"],
            save_criterion=full_config["learning"]["save_criterion"],
            mode=full_config["learning"]["mode"],
            early_stopping_patience=full_config["learning"]["early_stopping_patience"],
            log_interval=full_config["learning"]["log_interval"],
        )

        logger.info.assert_any_call("TRAINING STARTED")

    @patch("scripts.train.get_train_dataloader")
    @patch("scripts.train.get_val_dataloader")
    @patch("scripts.train.ProjectPaths")
    def test_train_loads_checkpoint(
        self,
        mock_project_paths_class,
        mock_get_val_loader,
        mock_get_train_loader,
        mock_paths,
        full_config,
        mock_logger,
        trained_checkpoint,
    ):
        checkpoint_path = mock_paths.SAVED_CHECKPOINTS / "test_model_best.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(trained_checkpoint.read_bytes())

        mock_project_paths_class.return_value = mock_paths
        logger = mock_logger

        mock_get_train_loader.return_value = MagicMock()
        mock_get_val_loader.return_value = MagicMock()

        with patch("scripts.train.create_model", return_value=MagicMock()), patch(
            "scripts.train.create_optimizer", return_value=MagicMock()
        ), patch("scripts.train.create_loss", return_value=MagicMock()), patch(
            "scripts.train.create_metrics", return_value={}
        ), patch(
            "scripts.train.create_scheduler", return_value=MagicMock()
        ), patch(
            "scripts.train.Trainer"
        ) as mock_trainer_class:
            mock_trainer = MagicMock()
            mock_trainer.current_epoch = 3
            mock_trainer_class.return_value = mock_trainer

            train(logger)

        mock_trainer.load_checkpoint.assert_called_once_with(
            path=checkpoint_path, load_optimizer=True, load_scheduler=True
        )
        logger.info.assert_any_call(
            "Resuming from epoch: %d", mock_trainer.current_epoch
        )

    @patch("scripts.train.get_train_dataloader")
    @patch("scripts.train.get_val_dataloader")
    @patch("scripts.train.ProjectPaths")
    def test_train_handles_keyboard_interrupt(
        self,
        mock_project_paths_class,
        mock_get_val_loader,
        mock_get_train_loader,
        mock_paths,
        full_config,
        mock_logger,
    ):
        mock_project_paths_class.return_value = mock_paths
        logger = mock_logger

        mock_get_train_loader.return_value = MagicMock()
        mock_get_val_loader.return_value = MagicMock()

        with patch("scripts.train.create_model", return_value=MagicMock()), patch(
            "scripts.train.create_optimizer", return_value=MagicMock()
        ), patch("scripts.train.create_loss", return_value=MagicMock()), patch(
            "scripts.train.create_metrics", return_value={}
        ), patch(
            "scripts.train.create_scheduler", return_value=MagicMock()
        ), patch(
            "scripts.train.Trainer"
        ) as mock_trainer_class:
            mock_trainer = MagicMock()
            mock_trainer.fit.side_effect = KeyboardInterrupt()
            mock_trainer_class.return_value = mock_trainer

            with pytest.raises(KeyboardInterrupt):
                train(logger)

        mock_trainer.save_checkpoint.assert_called_once_with(is_best=False)
        logger.info.assert_any_call("Training interrupted by user")

    @patch("scripts.train.get_train_dataloader")
    @patch("scripts.train.get_val_dataloader")
    @patch("scripts.train.ProjectPaths")
    def test_train_handles_exception_during_fit(
        self,
        mock_project_paths_class,
        mock_get_val_loader,
        mock_get_train_loader,
        mock_paths,
        full_config,
        mock_logger,
    ):
        mock_project_paths_class.return_value = mock_paths
        logger = mock_logger

        mock_get_train_loader.return_value = MagicMock()
        mock_get_val_loader.return_value = MagicMock()

        with patch("scripts.train.create_model", return_value=MagicMock()), patch(
            "scripts.train.create_optimizer", return_value=MagicMock()
        ), patch("scripts.train.create_loss", return_value=MagicMock()), patch(
            "scripts.train.create_metrics", return_value={}
        ), patch(
            "scripts.train.create_scheduler", return_value=MagicMock()
        ), patch(
            "scripts.train.Trainer"
        ) as mock_trainer_class:
            mock_trainer = MagicMock()
            mock_trainer.fit.side_effect = RuntimeError("Training error")
            mock_trainer_class.return_value = mock_trainer

            with pytest.raises(RuntimeError, match="Training error"):
                train(logger)

        logger.critical.assert_called_once_with(
            "Training failed. Check error log for details."
        )
        logger.exception.assert_called_once_with("Fatal error during training: %s", ANY)

    @patch("scripts.train.get_train_dataloader")
    @patch("scripts.train.get_val_dataloader")
    @patch("scripts.train.ProjectPaths")
    def test_train_prevent_sleep_windows(
        self,
        mock_project_paths_class,
        mock_get_val_loader,
        mock_get_train_loader,
        mock_paths,
        full_config,
        mock_logger,
        monkeypatch,
    ):
        mock_project_paths_class.return_value = mock_paths
        logger = mock_logger

        mock_get_train_loader.return_value = MagicMock()
        mock_get_val_loader.return_value = MagicMock()

        monkeypatch.setattr(platform, "system", lambda: "Windows")
        mock_set_thread_execution_state = MagicMock()
        monkeypatch.setattr(
            ctypes.windll.kernel32,
            "SetThreadExecutionState",
            mock_set_thread_execution_state,
        )

        with patch("scripts.train.create_model", return_value=MagicMock()), patch(
            "scripts.train.create_optimizer", return_value=MagicMock()
        ), patch("scripts.train.create_loss", return_value=MagicMock()), patch(
            "scripts.train.create_metrics", return_value={}
        ), patch(
            "scripts.train.create_scheduler", return_value=MagicMock()
        ), patch(
            "scripts.train.Trainer"
        ) as mock_trainer_class:
            mock_trainer = MagicMock()
            mock_trainer_class.return_value = mock_trainer

            train(logger)

        mock_set_thread_execution_state.assert_any_call(0x80000002)
        mock_set_thread_execution_state.assert_any_call(0x80000000)
        logger.info.assert_any_call("Sleep prevention activated (Windows)")
        logger.info.assert_any_call("Sleep prevention deactivated (Windows)")

    @patch("scripts.train.get_train_dataloader")
    @patch("scripts.train.get_val_dataloader")
    @patch("scripts.train.ProjectPaths")
    def test_train_handles_trainer_init_failure(
        self,
        mock_project_paths_class,
        mock_get_val_loader,
        mock_get_train_loader,
        mock_paths,
        full_config,
        mock_logger,
    ):
        mock_project_paths_class.return_value = mock_paths
        logger = mock_logger

        with patch("scripts.train.create_model", return_value=MagicMock()), patch(
            "scripts.train.create_optimizer", return_value=MagicMock()
        ), patch("scripts.train.create_loss", return_value=MagicMock()), patch(
            "scripts.train.create_metrics", return_value={}
        ), patch(
            "scripts.train.create_scheduler", return_value=MagicMock()
        ), patch(
            "scripts.train.Trainer", side_effect=Exception("Init failed")
        ):
            with pytest.raises(Exception, match="Init failed"):
                train(logger)

        logger.critical.assert_called_once_with(
            "Training failed. Check error log for details."
        )
        logger.exception.assert_called_once_with(
            "Trainer initialization failed: %s", ANY
        )

    @patch("scripts.train.get_train_dataloader")
    @patch("scripts.train.get_val_dataloader")
    @patch("scripts.train.ProjectPaths")
    def test_train_handles_checkpoint_load_failure(
        self,
        mock_project_paths_class,
        mock_get_val_loader,
        mock_get_train_loader,
        mock_paths,
        full_config,
        mock_logger,
        trained_checkpoint,
    ):
        checkpoint_path = mock_paths.SAVED_CHECKPOINTS / "test_model_best.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(trained_checkpoint.read_bytes())

        mock_project_paths_class.return_value = mock_paths
        logger = mock_logger

        mock_get_train_loader.return_value = MagicMock()
        mock_get_val_loader.return_value = MagicMock()

        with patch("scripts.train.create_model", return_value=MagicMock()), patch(
            "scripts.train.create_optimizer", return_value=MagicMock()
        ), patch("scripts.train.create_loss", return_value=MagicMock()), patch(
            "scripts.train.create_metrics", return_value={}
        ), patch(
            "scripts.train.create_scheduler", return_value=MagicMock()
        ), patch(
            "scripts.train.Trainer"
        ) as mock_trainer_class:
            mock_trainer = MagicMock()
            mock_trainer.load_checkpoint.side_effect = Exception("Load failed")
            mock_trainer_class.return_value = mock_trainer

            with pytest.raises(Exception, match="Load failed"):
                train(logger)

        logger.critical.assert_called_once_with(
            "Training failed. Check error log for details."
        )
        logger.exception.assert_called_once_with("Failed to load checkpoint: %s", ANY)
