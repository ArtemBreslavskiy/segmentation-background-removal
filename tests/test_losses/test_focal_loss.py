import numpy as np
import pytest
import torch

from src.losses.FocalLoss import FocalLoss


class TestFocalLoss:
    def test_focal_loss_returns_float(self):
        loss_function = FocalLoss(alpha=0.75, gamma=2.0)

        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()

        loss = loss_function(pred, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_focal_loss_perfect_match(self):
        loss_function = FocalLoss(alpha=0.75, gamma=2.0)

        pred_ones = torch.ones(2, 1, 64, 64) * 100
        target_ones = torch.ones(2, 1, 64, 64).float()

        pred_zeros = torch.ones(2, 1, 64, 64) * -100
        target_zeros = torch.zeros(2, 1, 64, 64).float()

        loss_one = loss_function(pred_ones, target_ones)
        loss_zero = loss_function(pred_zeros, target_zeros)

        assert loss_one.item() == pytest.approx(0.0, abs=1e-3)
        assert loss_zero.item() == pytest.approx(0.0, abs=1e-3)

    def test_focal_loss_perfect_mismatch(self):
        loss_function = FocalLoss(alpha=0.75, gamma=2.0)

        pred_ones = torch.ones(2, 1, 64, 64) * 100
        target_zeros = torch.zeros(2, 1, 64, 64).float()

        pred_zeros = torch.ones(2, 1, 64, 64) * -100
        target_ones = torch.ones(2, 1, 64, 64).float()

        loss_smatch = loss_function(pred_ones, target_ones)
        loss_mismatch_1 = loss_function(pred_ones, target_zeros)
        loss_mismatch_2 = loss_function(pred_zeros, target_ones)

        assert loss_mismatch_1.item() > loss_smatch
        assert loss_mismatch_2.item() > loss_smatch

    def test_focal_loss_gradients(self):
        loss_function = FocalLoss(alpha=0.75, gamma=2.0)

        pred = torch.randn(2, 1, 64, 64, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()

        loss = loss_function(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert not torch.all(pred.grad == 0)
        assert pred.grad.shape == pred.shape

    @pytest.mark.parametrize("batch_size", list(range(1, 33)))
    def test_focal_loss_different_batch_sizes(self, batch_size):
        loss_function = FocalLoss(alpha=0.75, gamma=2.0)

        pred = torch.randn(batch_size, 1, 64, 64)
        target = torch.randint(0, 2, (batch_size, 1, 64, 64)).float()

        loss = loss_function(pred, target)
        assert loss.dim() == 0
        assert loss.item() >= 0

    @pytest.mark.parametrize(
        "height, width",
        [(i, j) for i in range(32, 513, 32) for j in range(32, 513, 32)],
    )
    def test_focal_loss_different_resolutions(self, height, width):
        loss_function = FocalLoss(alpha=0.75, gamma=2.0)

        pred = torch.randn(2, 1, height, width)
        target = torch.randint(0, 2, (2, 1, height, width)).float()

        loss = loss_function(pred, target)

        assert loss.dim() == 0
        assert loss.item() >= 0

    @pytest.mark.parametrize("alpha", np.arange(0.05, 1, 0.05))
    def test_focal_loss_different_alpha(self, alpha):
        loss_function = FocalLoss(alpha=alpha, gamma=2.0)

        pred_ones = torch.ones(2, 1, 64, 64) * 100
        target_ones = torch.ones(2, 1, 64, 64).float()
        loss_one = loss_function(pred_ones, target_ones)

        pred_zeros = torch.ones(2, 1, 64, 64) * -100
        target_zeros = torch.zeros(2, 1, 64, 64).float()
        loss_zero = loss_function(pred_zeros, target_zeros)

        assert loss_zero.item() >= 0
        assert loss_one.item() >= 0
        assert loss_zero.item() == pytest.approx(0.0, abs=1e-3)
        assert loss_one.item() == pytest.approx(0.0, abs=1e-3)

    def test_focal_loss_alpha_monotonic(self):
        loss_functions = [FocalLoss(alpha=alpha, gamma=2.0) for alpha in np.arange(0.0, 1.0, 0.05)]

        pred_uncertain = torch.zeros(2, 1, 64, 64)
        target = torch.ones(2, 1, 64, 64).float()

        losses = [lf(pred_uncertain, target).item() for lf in loss_functions]

        for i in range(1, len(losses)):
            assert losses[i - 1] <= losses[i] + 1e-5

    def test_focal_loss_gamma_monotonic(self):
        loss_functions = [FocalLoss(alpha=0.75, gamma=gamma) for gamma in np.arange(0.0, 5.0, 0.1)]

        pred = torch.ones(2, 1, 64, 64) * 2
        target = torch.ones(2, 1, 64, 64).float()

        losses = [lf(pred, target).item() for lf in loss_functions]

        for i in range(1, len(losses)):
            assert losses[i] <= losses[i - 1] + 1e-5

    def test_focal_loss_extreme_value(self):
        loss_function = FocalLoss(alpha=0.75, gamma=2.0)

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

    def test_focal_loss_monotonic(self):
        loss_function = FocalLoss(alpha=0.75, gamma=2.0)

        preds = [torch.ones(2, 1, 64, 64) * v for v in range(-100, 101)]
        target_ones = torch.ones(2, 1, 64, 64).float()

        losses = [loss_function(p, target_ones).item() for p in preds]
        for i in range(1, len(losses)):
            assert losses[i] <= losses[i - 1] + 1e-5

    def test_focal_loss_reproducibility(self):
        loss_function = FocalLoss(alpha=0.75, gamma=2.0)

        torch.manual_seed(42)
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()

        loss1 = loss_function(pred, target)
        loss2 = loss_function(pred, target)

        assert loss1.item() == pytest.approx(loss2.item(), abs=1e-10)

    def test_focal_loss_compared_to_bce(self):
        focal_loss_function = FocalLoss(alpha=0.5, gamma=2.0)
        bce_loss_function = torch.nn.BCEWithLogitsLoss()

        preds = [torch.ones(2, 1, 64, 64) * v for v in range(-100, 101)]
        target_zeros = torch.zeros(2, 1, 64, 64).float()
        target_ones = torch.ones(2, 1, 64, 64).float()

        focal_loss_1 = [focal_loss_function(p, target_zeros).item() for p in preds]
        focal_loss_2 = [focal_loss_function(p, target_ones).item() for p in preds]
        bce_loss_1 = [bce_loss_function(p, target_zeros).item() for p in preds]
        bce_loss_2 = [bce_loss_function(p, target_ones).item() for p in preds]

        for fl, bce in zip(focal_loss_1, bce_loss_1):
            assert fl <= bce + 1e-5
        for fl, bce in zip(focal_loss_2, bce_loss_2):
            assert fl <= bce + 1e-5
