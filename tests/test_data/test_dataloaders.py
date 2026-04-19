from unittest.mock import MagicMock, patch

import pytest
import torch

from data.dataloaders import (
    get_dataloader,
    get_test_dataloader,
    get_train_dataloader,
    get_val_dataloader,
    seed_worker,
)


class TestDataLoaders:

    def test_seed_worker(self):
        with patch("torch.initial_seed", return_value=12345):
            with patch("random.seed") as mock_random_seed:
                with patch("numpy.random.seed") as mock_np_seed:
                    seed_worker(42)
                    expected_seed = 12345 % 2**32
                    mock_random_seed.assert_called_once_with(expected_seed)
                    mock_np_seed.assert_called_once_with(expected_seed)

    def test_get_dataloader_invalid_mode(self, full_config):
        with pytest.raises(ValueError, match="Unknown mode: invalid"):
            get_dataloader(full_config, "/fake/path", "invalid")

    @patch("data.dataloaders.get_train_dataset")
    @patch("data.dataloaders.get_val_dataset")
    @patch("data.dataloaders.get_test_dataset")
    def test_get_dataloader_train_mode(
        self, mock_get_test, mock_get_val, mock_get_train, full_config, mock_paths
    ):
        config: dict = full_config.copy()
        config["dataloader"] = config["dataloader"].copy()
        config["dataloader"]["num_workers"] = 2
        config["dataloader"]["pin_memory"] = True
        config["dataloader"]["persistent_workers"] = True
        config["dataloader"]["prefetch_factor"] = 3

        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_get_train.return_value = mock_dataset

        train_path = mock_paths.TRAIN
        loader = get_dataloader(config, train_path, "train")

        mock_get_train.assert_called_once_with(config, train_path)
        mock_get_val.assert_not_called()
        mock_get_test.assert_not_called()

        assert loader.batch_size == config["dataloader"]["batch_sizes"]["train"]
        assert loader.num_workers == 2
        assert loader.pin_memory is True
        assert loader.persistent_workers is True
        assert loader.prefetch_factor == 3
        assert loader.worker_init_fn == seed_worker
        assert loader.generator is not None
        assert loader.generator.initial_seed() == config["dataloader"]["seed"]
        assert isinstance(loader.sampler, torch.utils.data.RandomSampler)

    @patch("data.dataloaders.get_train_dataset")
    @patch("data.dataloaders.get_val_dataset")
    @patch("data.dataloaders.get_test_dataset")
    def test_get_dataloader_val_mode(
        self, mock_get_test, mock_get_val, mock_get_train, full_config, mock_paths
    ):
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_get_val.return_value = mock_dataset

        val_path = mock_paths.VAL
        loader = get_dataloader(full_config, val_path, "val")

        mock_get_val.assert_called_once_with(full_config, val_path)
        mock_get_train.assert_not_called()
        mock_get_test.assert_not_called()

        assert loader.batch_size == full_config["dataloader"]["batch_sizes"]["val"]
        assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)
        assert loader.generator is None

    @patch("data.dataloaders.get_train_dataset")
    @patch("data.dataloaders.get_val_dataset")
    @patch("data.dataloaders.get_test_dataset")
    def test_get_dataloader_test_mode(
        self, mock_get_test, mock_get_val, mock_get_train, full_config, mock_paths
    ):
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_get_test.return_value = mock_dataset

        test_path = mock_paths.TEST
        loader = get_dataloader(full_config, test_path, "test")

        mock_get_test.assert_called_once_with(full_config, test_path)
        mock_get_train.assert_not_called()
        mock_get_val.assert_not_called()

        assert loader.batch_size == full_config["dataloader"]["batch_sizes"]["test"]
        assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)
        assert loader.generator is None

    def test_get_dataloader_with_string_path(self, full_config):
        path = "/some/path"
        with patch("data.dataloaders.get_train_dataset") as mock_get_train:
            mock_dataset = MagicMock()
            mock_dataset.__len__.return_value = 100
            mock_get_train.return_value = mock_dataset

            mock_get_train.assert_called_once_with(full_config, path)

    def test_get_dataloader_with_num_workers(self, full_config):
        config: dict = full_config.copy()
        config["dataloader"] = config["dataloader"].copy()
        config["dataloader"]["num_workers"] = 4
        config["dataloader"]["pin_memory"] = True
        config["dataloader"]["persistent_workers"] = True
        config["dataloader"]["prefetch_factor"] = 3

        with patch("data.dataloaders.get_train_dataset") as mock_get_train:
            mock_dataset = MagicMock()
            mock_dataset.__len__.return_value = 100
            mock_get_train.return_value = mock_dataset

            loader = get_dataloader(config, "/fake/path", "train")

            assert loader.num_workers == 4
            assert loader.pin_memory is True
            assert loader.persistent_workers is True
            assert loader.prefetch_factor == 3
            assert loader.worker_init_fn == seed_worker

    def test_get_train_dataloader(self, full_config, mock_paths):
        with patch("data.dataloaders.get_dataloader") as mock_get_dataloader:
            mock_get_dataloader.return_value = MagicMock()
            result = get_train_dataloader(full_config, mock_paths)
            mock_get_dataloader.assert_called_once_with(
                config=full_config, path=mock_paths, mode="train"
            )
            assert result == mock_get_dataloader.return_value

    def test_get_val_dataloader(self, full_config, mock_paths):
        with patch("data.dataloaders.get_dataloader") as mock_get_dataloader:
            mock_get_dataloader.return_value = MagicMock()
            result = get_val_dataloader(full_config, mock_paths)
            mock_get_dataloader.assert_called_once_with(
                config=full_config, path=mock_paths, mode="val"
            )
            assert result == mock_get_dataloader.return_value

    def test_get_test_dataloader(self, full_config, mock_paths):
        with patch("data.dataloaders.get_dataloader") as mock_get_dataloader:
            mock_get_dataloader.return_value = MagicMock()
            result = get_test_dataloader(full_config, mock_paths)
            mock_get_dataloader.assert_called_once_with(
                config=full_config, path=mock_paths, mode="test"
            )
            assert result == mock_get_dataloader.return_value
