import pytest
import torch
import torch.nn as nn

from src.losses.ComboLoss import ComboLoss
from src.losses.FocalLoss import FocalLoss
from src.losses.SoftDiceLoss import SoftDiceLoss


class MockLoss(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, pred, target):
        return torch.tensor(self.value, dtype=torch.float32)


class TestComboLoss:
    def test_combo_loss_returns_float(self):
        loss_function = ComboLoss(
            [SoftDiceLoss(smooth=1.0), FocalLoss(alpha=0.75, gamma=2.0)], [0.7, 0.3]
        )

        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()

        loss = loss_function(pred, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_combo_loss_perfect_match(self):
        loss_function = ComboLoss(
            [SoftDiceLoss(smooth=1.0), FocalLoss(alpha=0.75, gamma=2.0)], [0.7, 0.3]
        )

        pred_ones = torch.ones(2, 1, 64, 64) * 100
        target_ones = torch.ones(2, 1, 64, 64).float()

        pred_zeros = torch.ones(2, 1, 64, 64) * -100
        target_zeros = torch.zeros(2, 1, 64, 64).float()

        loss_one = loss_function(pred_ones, target_ones)
        loss_zero = loss_function(pred_zeros, target_zeros)

        assert loss_one.item() == pytest.approx(0.0, abs=1e-3)
        assert loss_zero.item() == pytest.approx(0.0, abs=1e-3)

    def test_combo_loss_perfect_mismatch(self):
        loss_function = ComboLoss(
            [SoftDiceLoss(smooth=1.0), FocalLoss(alpha=0.75, gamma=2.0)], [0.7, 0.3]
        )

        pred_ones = torch.ones(2, 1, 64, 64) * 100
        target_zeros = torch.zeros(2, 1, 64, 64).float()

        pred_zeros = torch.ones(2, 1, 64, 64) * -100
        target_ones = torch.ones(2, 1, 64, 64).float()

        loss_smatch = loss_function(pred_ones, target_ones)
        loss_mismatch_1 = loss_function(pred_ones, target_zeros)
        loss_mismatch_2 = loss_function(pred_zeros, target_ones)

        assert loss_mismatch_1.item() > loss_smatch
        assert loss_mismatch_2.item() > loss_smatch

    def test_combo_loss_gradients(self):
        loss_function = ComboLoss(
            [SoftDiceLoss(smooth=1.0), FocalLoss(alpha=0.75, gamma=2.0)], [0.7, 0.3]
        )

        pred = torch.randn(2, 1, 64, 64, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()

        loss = loss_function(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert not torch.all(pred.grad == 0)
        assert pred.grad.shape == pred.shape

    def test_combo_loss_gradients_with_components(self):
        loss_function = ComboLoss(
            [SoftDiceLoss(smooth=1.0), FocalLoss(alpha=0.75, gamma=2.0)], [0.7, 0.3]
        )

        pred = torch.randn(2, 1, 64, 64, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()

        total_loss, components = loss_function.forward_with_components(pred, target)
        total_loss.backward()

        assert pred.grad is not None
        assert not torch.all(pred.grad == 0)
        assert pred.grad.shape == pred.shape

        for c in components.values():
            assert c.grad_fn is not None

    @pytest.mark.parametrize("batch_size", list(range(1, 33)))
    def test_combo_loss_different_batch_sizes(self, batch_size):
        loss_function = ComboLoss(
            [SoftDiceLoss(smooth=1.0), FocalLoss(alpha=0.75, gamma=2.0)], [0.7, 0.3]
        )

        pred = torch.randn(batch_size, 1, 64, 64)
        target = torch.randint(0, 2, (batch_size, 1, 64, 64)).float()

        loss = loss_function(pred, target)
        assert loss.dim() == 0
        assert loss.item() >= 0

    @pytest.mark.parametrize(
        "height, width",
        [(i, j) for i in range(32, 513, 32) for j in range(32, 513, 32)],
    )
    def test_combo_loss_different_resolutions(self, height, width):
        loss_function = ComboLoss(
            [SoftDiceLoss(smooth=1.0), FocalLoss(alpha=0.75, gamma=2.0)], [0.7, 0.3]
        )

        pred = torch.randn(2, 1, height, width)
        target = torch.randint(0, 2, (2, 1, height, width)).float()

        loss = loss_function(pred, target)

        assert loss.dim() == 0
        assert loss.item() >= 0

    @pytest.mark.parametrize("num_losses", list(range(1, 33)))
    def test_combo_loss_different_number_of_losses(self, num_losses):
        losses = [MockLoss(0.5) for _ in range(num_losses)]
        combo_function = ComboLoss(losses)

        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()

        loss = combo_function(pred, target)

        assert loss.dim() == 0
        assert loss.item() >= 0
        assert combo_function.count == num_losses

    def test_combo_loss_extreme_value(self):
        loss_function = ComboLoss(
            [SoftDiceLoss(smooth=1.0), FocalLoss(alpha=0.75, gamma=2.0)], [0.7, 0.3]
        )

        pred_big = torch.ones(2, 1, 64, 64) * 1e6
        pred_small = torch.ones(2, 1, 64, 64) * 1e-6

        target_ones = torch.ones(2, 1, 64, 64).float()
        target_zeros = torch.zeros(2, 1, 64, 64).float()

        loss_zero_1 = loss_function(pred_big, target_zeros)
        loss_zero_2 = loss_function(pred_small, target_ones)
        loss_one_1 = loss_function(pred_big, target_ones)
        loss_one_2 = loss_function(pred_small, target_zeros)

        assert torch.isfinite(loss_zero_1)
        assert torch.isfinite(loss_zero_2)
        assert torch.isfinite(loss_one_1)
        assert torch.isfinite(loss_one_2)

        assert loss_zero_1.item() >= 0
        assert loss_zero_2.item() >= 0
        assert loss_one_1.item() >= 0
        assert loss_one_2.item() >= 0

    def test_combo_loss_monotonic(self):
        loss_function = ComboLoss(
            [SoftDiceLoss(smooth=1.0), FocalLoss(alpha=0.75, gamma=2.0)], [0.7, 0.3]
        )

        preds = [torch.ones(2, 1, 64, 64) * v for v in range(-100, 101)]
        target_ones = torch.ones(2, 1, 64, 64).float()

        losses = [loss_function(p, target_ones).item() for p in preds]
        for i in range(1, len(losses)):
            assert losses[i] <= losses[i - 1] + 1e-5

    def test_combo_loss_reproducibility(self):
        loss_function = ComboLoss(
            [SoftDiceLoss(smooth=1.0), FocalLoss(alpha=0.75, gamma=2.0)], [0.7, 0.3]
        )

        torch.manual_seed(42)
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()

        loss1 = loss_function(pred, target)
        loss2 = loss_function(pred, target)

        assert loss1.item() == pytest.approx(loss2.item(), abs=1e-10)

    def test_combo_loss_initialization_empty(self):
        with pytest.raises(ValueError, match="loss_functions cannot be empty"):
            ComboLoss([])

    def test_combo_loss_initialization_wrong_weights_length(self):
        with pytest.raises(ValueError, match="Number of weights must match"):
            ComboLoss(
                [SoftDiceLoss(smooth=1.0), FocalLoss(alpha=0.75, gamma=2.0)], [0.7]
            )

    def test_combo_loss_initialization_negative_weights_sum(self):
        with pytest.raises(ValueError, match="Sum of weights must be positive"):
            ComboLoss(
                [SoftDiceLoss(smooth=1.0), FocalLoss(alpha=0.75, gamma=2.0)],
                [-0.7, 0.3],
            )

    def test_combo_loss_initialization_without_weights(self):
        loss_function = ComboLoss(
            [SoftDiceLoss(smooth=1.0), FocalLoss(alpha=0.75, gamma=2.0)],
        )

        assert len(loss_function.weights) == 2
        assert loss_function.weights[0] == pytest.approx(0.5, abs=1e-5)
        assert loss_function.weights[1] == pytest.approx(0.5, abs=1e-5)

    def test_combo_loss_weights_normalization(self):
        weights = [1.0, 2.0, 3.0]
        expected_weights = [1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0]

        loss_function = ComboLoss(
            [
                SoftDiceLoss(smooth=1.0),
                FocalLoss(alpha=0.75, gamma=2.0),
                SoftDiceLoss(smooth=0.5),
            ],
            weights,
        )

        for actual, expected in zip(loss_function.weights, expected_weights):
            assert actual == pytest.approx(expected, abs=1e-5)

    def test_combo_loss_calculation(self):
        loss_function = ComboLoss(
            [MockLoss(0.5), MockLoss(0.3), MockLoss(0.2)], [1.0, 2.0, 3.0]
        )

        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()

        loss = loss_function(pred, target)
        expected_loss = (0.5 * 1 / 6) + (0.3 * 2 / 6) + (0.2 * 3 / 6)

        assert loss.item() == pytest.approx(expected_loss, abs=1e-5)

    def test_combo_loss_forward_with_components(self):
        soft_dice_loss_function = SoftDiceLoss(smooth=1.0)
        focal_loss_function = FocalLoss(alpha=0.75, gamma=2.0)
        loss_functions = [soft_dice_loss_function, focal_loss_function]
        weights = [0.3, 0.7]
        combo_function = ComboLoss(loss_functions, weights)

        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()

        total_loss, components = combo_function.forward_with_components(pred, target)
        soft_dice_loss = soft_dice_loss_function(pred, target)
        focal_loss = focal_loss_function(pred, target)

        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(components, dict)
        assert "SoftDiceLoss" in components
        assert "FocalLoss" in components
        assert isinstance(components["SoftDiceLoss"], torch.Tensor)
        assert isinstance(components["FocalLoss"], torch.Tensor)

        assert total_loss >= 0
        assert components["SoftDiceLoss"].item() >= 0
        assert components["FocalLoss"].item() >= 0

        assert len(loss_functions) == len(components)

        weighted = (
            components["SoftDiceLoss"].item() * combo_function.weights[0]
            + components["FocalLoss"].item() * combo_function.weights[1]
        )
        assert total_loss == pytest.approx(weighted, abs=1e-5)

        weighted = (
            soft_dice_loss * combo_function.weights[0]
            + focal_loss * combo_function.weights[1]
        )
        assert total_loss == pytest.approx(weighted, abs=1e-5)

    def test_combo_loss_get_raw_losses(self):
        soft_dice_loss_function = SoftDiceLoss(smooth=1.0)
        focal_loss_function = FocalLoss(alpha=0.75, gamma=2.0)
        loss_functions = [soft_dice_loss_function, focal_loss_function]
        weights = [0.3, 0.7]
        combo_function = ComboLoss(loss_functions, weights)

        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()

        raw_losses = combo_function.get_raw_losses(pred, target)
        soft_dice_loss = soft_dice_loss_function(pred, target)
        focal_loss = focal_loss_function(pred, target)

        assert isinstance(raw_losses, dict)
        assert "SoftDiceLoss" in raw_losses
        assert "FocalLoss" in raw_losses
        assert isinstance(raw_losses["SoftDiceLoss"], torch.Tensor)
        assert isinstance(raw_losses["FocalLoss"], torch.Tensor)

        assert len(raw_losses) == len(loss_functions)

        assert raw_losses["SoftDiceLoss"].item() == pytest.approx(
            soft_dice_loss, abs=1e-5
        )
        assert raw_losses["FocalLoss"].item() == pytest.approx(focal_loss, abs=1e-5)

        assert raw_losses["SoftDiceLoss"].item() >= 0
        assert raw_losses["FocalLoss"].item() >= 0

    def test_combo_loss_names_property(self):
        loss_functions = [
            SoftDiceLoss(smooth=1.0),
            FocalLoss(alpha=0.75, gamma=2.0),
            SoftDiceLoss(smooth=0.5),
        ]
        combo_function = ComboLoss(loss_functions)

        assert len(combo_function.names) == len(loss_functions)
        assert combo_function.names == ["SoftDiceLoss", "FocalLoss", "SoftDiceLoss_1"]

    def test_combo_loss_get_weights_property(self):
        loss_functions = [
            SoftDiceLoss(smooth=1.0),
            FocalLoss(alpha=0.75, gamma=2.0),
            SoftDiceLoss(smooth=0.5),
        ]
        weights = [0.1, 0.3, 0.6]
        combo_function = ComboLoss(loss_functions, weights)

        assert len(combo_function.get_weights) == len(weights)
        for actual, expected in zip(combo_function.weights, weights):
            assert actual == pytest.approx(expected, abs=1e-5)

    def test_combo_loss_count_property(self):
        loss_functions = [
            SoftDiceLoss(smooth=1.0),
            FocalLoss(alpha=0.75, gamma=2.0),
            SoftDiceLoss(smooth=0.5),
        ]
        combo_function = ComboLoss(loss_functions)

        assert combo_function.count == len(loss_functions)
