from unittest.mock import ANY, MagicMock, patch

import pytest

from src.utils.factory import (
    _convert_value,
    _get_class,
    create_loss,
    create_metrics,
    create_model,
    create_optimizer,
    create_scheduler,
)


class TestFactory:

    # ---------- _get_class ----------
    def test_get_class(self):
        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_class = MagicMock()
            mock_module.TestClass = mock_class
            mock_import.return_value = mock_module

            result = _get_class("some.module.TestClass")

            mock_import.assert_called_once_with("some.module")
            assert result == mock_class

    def test_get_class_invalid_path(self):
        with patch("importlib.import_module", side_effect=ImportError):
            with pytest.raises(ImportError):
                _get_class("invalid.path")

    def test_convert_value_int(self):
        assert _convert_value("123") == 123
        assert _convert_value("-456") == -456
        assert _convert_value("0") == 0

    def test_convert_value_float(self):
        assert _convert_value("123.45") == 123.45
        assert _convert_value("-67.89") == -67.89
        assert _convert_value("100.0") == 100.0

    def test_convert_value_non_numeric(self):
        assert _convert_value("abc") == "abc"
        assert _convert_value("123abc") == "123abc"

    def test_convert_value_dict(self):
        input_dict = {"a": "10", "b": {"c": "20.5", "d": "xyz"}}
        expected = {"a": 10, "b": {"c": 20.5, "d": "xyz"}}
        assert _convert_value(input_dict) == expected

    def test_convert_value_list(self):
        input_list = ["1", "2.5", "abc"]
        expected = [1, 2.5, "abc"]
        assert _convert_value(input_list) == expected

    def test_convert_value_nested(self):
        input_data = {
            "params": {
                "lr": "0.001",
                "batch_size": "32",
                "scheduler": {"step_size": "10"},
            }
        }
        expected = {
            "params": {"lr": 0.001, "batch_size": 32, "scheduler": {"step_size": 10}}
        }
        assert _convert_value(input_data) == expected

    @patch("src.utils.factory._get_class")
    def test_create_model(self, mock_get_class, full_config):
        mock_class = MagicMock()
        mock_get_class.return_value = mock_class

        result = create_model(full_config)

        mock_get_class.assert_called_once_with(full_config["model"]["class"])
        expected_params = _convert_value(full_config["model"]["params"])
        mock_class.assert_called_once_with(**expected_params)
        assert result == mock_class.return_value

    @patch("src.utils.factory._get_class")
    def test_create_optimizer(self, mock_get_class, full_config, model):
        mock_class = MagicMock()
        mock_get_class.return_value = mock_class

        result = create_optimizer(full_config, model)

        mock_get_class.assert_called_once_with(
            full_config["learning"]["optimizer"]["class"]
        )
        expected_params = _convert_value(full_config["learning"]["optimizer"]["params"])
        mock_class.assert_called_once_with(ANY, **expected_params)
        assert result == mock_class.return_value

    @patch("src.utils.factory._get_class")
    def test_create_loss_standard(self, mock_get_class, full_config):
        mock_class = MagicMock()
        mock_get_class.return_value = mock_class

        result = create_loss(full_config)

        mock_get_class.assert_called_once_with(full_config["learning"]["loss"]["class"])
        expected_params = _convert_value(full_config["learning"]["loss"]["params"])
        mock_class.assert_called_once_with(**expected_params)
        assert result == mock_class.return_value

    @patch("src.losses.ComboLoss.ComboLoss")
    @patch("src.utils.factory._get_class")
    def test_create_loss_combo(self, mock_get_class, mock_combo_class, full_config):
        config = full_config.copy()
        config["learning"] = full_config["learning"].copy()
        config["learning"]["loss"] = {
            "class": "src.losses.ComboLoss.ComboLoss",
            "params": {
                "loss_functions": [
                    {"class": "torch.nn.BCELoss", "params": {}},
                    {"class": "torch.nn.DiceLoss", "params": {"smooth": "1e-5"}},
                ],
                "weights": [0.5, 0.5],
            },
        }

        mock_loss1 = MagicMock()
        mock_loss2 = MagicMock()
        mock_get_class.side_effect = [mock_loss1, mock_loss2]

        result = create_loss(config)

        assert mock_get_class.call_count == 2
        mock_get_class.assert_any_call("torch.nn.BCELoss")
        mock_get_class.assert_any_call("torch.nn.DiceLoss")

        mock_combo_class.assert_called_once_with(
            loss_functions=[mock_loss1.return_value, mock_loss2.return_value],
            weights=[0.5, 0.5],
        )
        assert result == mock_combo_class.return_value

    @patch("src.utils.factory._get_class")
    def test_create_metrics(self, mock_get_class, full_config):
        mock_metric1 = MagicMock()
        mock_metric2 = MagicMock()
        mock_get_class.side_effect = [mock_metric1, mock_metric2]

        result = create_metrics(full_config)

        assert len(result) == 2
        assert result["accuracy"] == mock_metric1.return_value
        assert result["iou"] == mock_metric2.return_value

        expected_params1 = _convert_value(
            full_config["evaluating"]["metrics"][0]["params"]
        )
        expected_params2 = _convert_value(
            full_config["evaluating"]["metrics"][1]["params"]
        )

        mock_metric1.assert_called_once_with(**expected_params1)
        mock_metric2.assert_called_once_with(**expected_params2)

    @patch("src.utils.factory._get_class")
    def test_create_scheduler(self, mock_get_class, full_config, optimizer):
        mock_scheduler_class = MagicMock()
        mock_get_class.return_value = mock_scheduler_class

        result = create_scheduler(full_config, optimizer)

        mock_get_class.assert_called_once_with(
            full_config["learning"]["scheduler"]["class"]
        )
        expected_params = _convert_value(full_config["learning"]["scheduler"]["params"])
        mock_scheduler_class.assert_called_once_with(optimizer, **expected_params)
        assert result == mock_scheduler_class.return_value
