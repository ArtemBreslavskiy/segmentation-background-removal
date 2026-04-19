from unittest.mock import patch

import pytest

from scripts.build_dataset import build_processed_dataset


class TestBuildDataset:
    @patch("scripts.build_dataset.ProjectPaths")
    @patch("builtins.input", return_value="y")
    @patch("shutil.copy2")
    @patch("sklearn.model_selection.train_test_split")
    def test_build_processed_dataset_success(
        self,
        mock_split,
        mock_copy2,
        mock_input,
        mock_project_paths_class,
        sample_files,
        mock_logger,
    ):
        mock_project_paths_class.return_value = sample_files
        logger = mock_logger

        all_images = sorted(
            sample_files.DUTS_TR_IMAGES.glob("*.jpg"), key=lambda x: x.stem
        )
        all_masks = sorted(
            sample_files.DUTS_TR_MASKS.glob("*.png"), key=lambda x: x.stem
        )
        train_images = all_images[:8]
        val_images = all_images[8:]
        train_masks = all_masks[:8]
        val_masks = all_masks[8:]
        mock_split.return_value = (train_images, val_images, train_masks, val_masks)

        build_processed_dataset(logger)

        logger.info.assert_any_call("=" * 60)
        logger.info.assert_any_call("BUILDING PROCESSED DATASET")
        logger.info.assert_any_call("Raw data path: %s", sample_files.RAW_DATA)
        logger.info.assert_any_call(
            "Processed data path: %s", sample_files.PROCESSED_DATA
        )

        total_files = (
            len(train_images) * 2
            + len(val_images) * 2
            + len(list(sample_files.DUTS_TE_IMAGES.glob("*.jpg"))) * 2
        )
        assert mock_copy2.call_count == total_files

        assert sample_files.TRAIN_IMAGES.exists()
        assert sample_files.TRAIN_MASKS.exists()
        assert sample_files.VAL_IMAGES.exists()
        assert sample_files.VAL_MASKS.exists()
        assert sample_files.TEST_IMAGES.exists()
        assert sample_files.TEST_MASKS.exists()

        logger.info.assert_any_call("DATASET BUILD COMPLETED SUCCESSFULLY")

    @patch("scripts.build_dataset.ProjectPaths")
    def test_build_processed_dataset_raw_not_exists(
        self, mock_project_paths_class, mock_paths, mock_logger
    ):
        mock_paths.RAW_DATA = mock_paths.base / "raw"
        mock_project_paths_class.return_value = mock_paths
        logger = mock_logger
        with pytest.raises(ValueError, match="Raw data not found"):
            build_processed_dataset(logger)
        logger.exception.assert_called_once()
        logger.critical.assert_called_once_with(
            "Dataset building failed. Check error log for details."
        )

    @patch("scripts.build_dataset.ProjectPaths")
    @patch("builtins.input", return_value="n")
    def test_build_processed_dataset_processed_exists_cancel(
        self, mock_input, mock_project_paths_class, mock_paths, mock_logger
    ):
        mock_paths.RAW_DATA.mkdir(parents=True, exist_ok=True)
        mock_paths.PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
        mock_project_paths_class.return_value = mock_paths
        logger = mock_logger

        build_processed_dataset(logger)
        mock_input.assert_called_once()
        assert mock_paths.PROCESSED_DATA.exists()
        logger.info.assert_any_call("Operation cancelled by user")
        assert not mock_paths.TRAIN_IMAGES.exists()

    @patch("scripts.build_dataset.ProjectPaths")
    @patch("shutil.rmtree")
    @patch("builtins.input", return_value="y")
    @patch("shutil.copy2")
    @patch("sklearn.model_selection.train_test_split")
    def test_build_processed_dataset_processed_exists_yes(
        self,
        mock_split,
        mock_copy2,
        mock_input,
        mock_rmtree,
        mock_project_paths_class,
        sample_files,
        mock_logger,
    ):
        sample_files.PROCESSED_DATA.mkdir(parents=True)
        mock_project_paths_class.return_value = sample_files

        all_images = sorted(
            sample_files.DUTS_TR_IMAGES.glob("*.jpg"), key=lambda x: x.stem
        )
        all_masks = sorted(
            sample_files.DUTS_TR_MASKS.glob("*.png"), key=lambda x: x.stem
        )
        train_images = all_images[:8]
        val_images = all_images[8:]
        train_masks = all_masks[:8]
        val_masks = all_masks[8:]
        mock_split.return_value = (train_images, val_images, train_masks, val_masks)

        logger = mock_logger
        build_processed_dataset(logger)

        mock_rmtree.assert_called_once_with(sample_files.PROCESSED_DATA)
        assert mock_copy2.call_count > 0
        logger.info.assert_any_call("Cleaning existing processed data...")

    @patch("scripts.build_dataset.ProjectPaths")
    @patch("builtins.input", return_value="y")
    def test_build_processed_dataset_mismatch_train_count(
        self, mock_input, mock_project_paths_class, sample_files, mock_logger
    ):
        (sample_files.DUTS_TR_MASKS / "img9.png").unlink()
        mock_project_paths_class.return_value = sample_files
        logger = mock_logger

        with pytest.raises(ValueError, match="Mismatch in train/val dataset"):
            build_processed_dataset(logger)

        logger.exception.assert_any_call(
            "Mismatch in train/val dataset: 10 images vs 9 masks"
        )
        assert logger.critical.call_count == 2
        logger.critical.assert_any_call(
            "Dataset building failed. Check error log for details."
        )

    @patch("scripts.build_dataset.ProjectPaths")
    @patch("builtins.input", return_value="y")
    def test_build_processed_dataset_mismatch_test_count(
        self, mock_input, mock_project_paths_class, sample_files, mock_logger
    ):
        (sample_files.DUTS_TE_MASKS / "test0.png").unlink()
        mock_project_paths_class.return_value = sample_files
        logger = mock_logger

        with pytest.raises(ValueError, match="Mismatch in test dataset"):
            build_processed_dataset(logger)

        logger.exception.assert_any_call(
            "Mismatch in test dataset: 5 images vs 4 masks"
        )
        assert logger.critical.call_count == 2
        logger.critical.assert_any_call(
            "Dataset building failed. Check error log for details."
        )

    @patch("scripts.build_dataset.ProjectPaths")
    @patch("builtins.input", return_value="y")
    def test_build_processed_dataset_filename_mismatch(
        self, mock_input, mock_project_paths_class, sample_files, mock_logger
    ):
        (sample_files.DUTS_TR_MASKS / "img9.png").rename(
            sample_files.DUTS_TR_MASKS / "wrong.png"
        )
        mock_project_paths_class.return_value = sample_files
        logger = mock_logger

        with pytest.raises(
            ValueError, match="Filename inconsistencies found in dataset"
        ):
            build_processed_dataset(logger)

        logger.error.assert_called()
        logger.warning.assert_called()

    @patch("scripts.build_dataset.ProjectPaths")
    @patch("builtins.input", return_value="y")
    def test_build_processed_dataset_empty_config(
        self, mock_input, mock_project_paths_class, sample_files, mock_logger
    ):
        sample_files.CONFIG.write_text("{}")
        mock_project_paths_class.return_value = sample_files
        logger = mock_logger

        with pytest.raises(KeyError):
            build_processed_dataset(logger)
        logger.exception.assert_called_once()

    @patch("scripts.build_dataset.ProjectPaths")
    @patch("builtins.input", return_value="y")
    def test_build_processed_dataset_exception_handling(
        self, mock_input, mock_project_paths_class, mock_paths, mock_logger
    ):
        mock_paths.RAW_DATA.mkdir(parents=True, exist_ok=True)
        mock_project_paths_class.return_value = mock_paths
        logger = mock_logger

        if mock_paths.CONFIG.exists():
            mock_paths.CONFIG.unlink()

        with pytest.raises(FileNotFoundError):
            build_processed_dataset(logger)

        logger.exception.assert_called_once_with("Error building dataset")
        logger.critical.assert_called_once_with(
            "Dataset building failed. Check error log for details."
        )
