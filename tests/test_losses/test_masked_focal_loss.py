import numpy as np
import pytest
import torch

from src.losses.MaskedFocalLoss import MaskedFocalLoss


class TestMaskedFocalLoss:
    def test_returns_float(self):
        loss_function = MaskedFocalLoss(alpha=0.75, gamma=2.0)

        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()

        loss = loss_function(pred, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_perfect_match(self):
        loss_function = MaskedFocalLoss(alpha=0.75, gamma=2.0)

        pred_ones = torch.ones(2, 1, 64, 64) * 100
        target_ones = torch.ones(2, 1, 64, 64).float()

        pred_zeros = torch.ones(2, 1, 64, 64) * -100
        target_zeros = torch.zeros(2, 1, 64, 64).float()

        loss_one = loss_function(pred_ones, target_ones)
        loss_zero = loss_function(pred_zeros, target_zeros)

        assert loss_one.item() == pytest.approx(0.0, abs=1e-3)
        assert loss_zero.item() == pytest.approx(0.0, abs=1e-3)

    def test_perfect_mismatch(self):
        loss_function = MaskedFocalLoss(alpha=0.75, gamma=2.0)

        pred_ones = torch.ones(2, 1, 64, 64) * 100
        target_zeros = torch.zeros(2, 1, 64, 64).float()

        pred_zeros = torch.ones(2, 1, 64, 64) * -100
        target_ones = torch.ones(2, 1, 64, 64).float()

        loss_smatch = loss_function(pred_ones, target_ones)
        loss_mismatch_1 = loss_function(pred_ones, target_zeros)
        loss_mismatch_2 = loss_function(pred_zeros, target_ones)

        assert loss_mismatch_1.item() > loss_smatch
        assert loss_mismatch_2.item() > loss_smatch

    def test_gradients(self):
        loss_function = MaskedFocalLoss(alpha=0.75, gamma=2.0)

        pred = torch.randn(2, 1, 64, 64, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()

        loss = loss_function(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert not torch.all(pred.grad == 0)
        assert pred.grad.shape == pred.shape

    @pytest.mark.parametrize("batch_size", list(range(1, 33)))
    def test_different_batch_sizes(self, batch_size):
        loss_function = MaskedFocalLoss(alpha=0.75, gamma=2.0)

        pred = torch.randn(batch_size, 1, 64, 64)
        target = torch.randint(0, 2, (batch_size, 1, 64, 64)).float()

        loss = loss_function(pred, target)
        assert loss.dim() == 0
        assert loss.item() >= 0

    @pytest.mark.parametrize(
        "height, width",
        [(i, j) for i in range(32, 513, 32) for j in range(32, 513, 32)],
    )
    def test_different_resolutions(self, height, width):
        loss_function = MaskedFocalLoss(alpha=0.75, gamma=2.0)

        pred = torch.randn(2, 1, height, width)
        target = torch.randint(0, 2, (2, 1, height, width)).float()

        loss = loss_function(pred, target)

        assert loss.dim() == 0
        assert loss.item() >= 0

    @pytest.mark.parametrize("alpha", np.arange(0.05, 1, 0.05))
    def test_different_alpha(self, alpha):
        loss_function = MaskedFocalLoss(alpha=alpha, gamma=2.0)

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

    def test_alpha_monotonic(self):
        loss_functions = [MaskedFocalLoss(alpha=alpha, gamma=2.0) for alpha in np.arange(0.0, 1.0, 0.05)]

        pred_uncertain = torch.zeros(2, 1, 64, 64)
        target = torch.ones(2, 1, 64, 64).float()

        losses = [lf(pred_uncertain, target).item() for lf in loss_functions]

        for i in range(1, len(losses)):
            assert losses[i - 1] <= losses[i] + 1e-5

    def test_gamma_monotonic(self):
        loss_functions = [MaskedFocalLoss(alpha=0.75, gamma=gamma) for gamma in np.arange(0.0, 5.0, 0.1)]

        pred = torch.ones(2, 1, 64, 64) * 2
        target = torch.ones(2, 1, 64, 64).float()

        losses = [lf(pred, target).item() for lf in loss_functions]

        for i in range(1, len(losses)):
            assert losses[i] <= losses[i - 1] + 1e-5

    def test_extreme_value(self):
        loss_function = MaskedFocalLoss(alpha=0.75, gamma=2.0)

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

    def test_monotonic(self):
        loss_function = MaskedFocalLoss(alpha=0.75, gamma=2.0)

        preds = [torch.ones(2, 1, 64, 64) * v for v in range(-100, 101)]
        target_ones = torch.ones(2, 1, 64, 64).float()

        losses = [loss_function(p, target_ones).item() for p in preds]
        for i in range(1, len(losses)):
            assert losses[i] <= losses[i - 1] + 1e-5

    def test_reproducibility(self):
        loss_function = MaskedFocalLoss(alpha=0.75, gamma=2.0)

        torch.manual_seed(42)
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()

        loss1 = loss_function(pred, target)
        loss2 = loss_function(pred, target)

        assert loss1.item() == pytest.approx(loss2.item(), abs=1e-10)

    def test_compared_to_bce(self):
        focal_loss_function = MaskedFocalLoss(alpha=0.5, gamma=2.0)
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

    def test_returns_float_with_valid_mask(self):
        loss_function = MaskedFocalLoss(alpha=0.75, gamma=2.0)

        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        valid_mask = torch.zeros(2, 1, 64, 64)
        valid_mask[:, :, :32, :] = 1.0

        loss = loss_function(pred, target, valid_mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_perfect_match_with_valid_mask(self):
        loss_function = MaskedFocalLoss(alpha=0.75, gamma=2.0)

        pred_ones = torch.ones(2, 1, 64, 64) * 100
        target_ones = torch.ones(2, 1, 64, 64).float()

        pred_zeros = torch.ones(2, 1, 64, 64) * -100
        target_zeros = torch.zeros(2, 1, 64, 64).float()

        valid_mask = torch.zeros(2, 1, 64, 64)
        valid_mask[:, :, :32, :] = 1.0

        loss_one = loss_function(pred_ones, target_ones, valid_mask)
        loss_zero = loss_function(pred_zeros, target_zeros, valid_mask)

        assert loss_one.item() == pytest.approx(0.0, abs=1e-3)
        assert loss_zero.item() == pytest.approx(0.0, abs=1e-3)

    def test_perfect_mismatch_with_valid_mask(self):
        loss_function = MaskedFocalLoss(alpha=0.75, gamma=2.0)

        pred_ones = torch.ones(2, 1, 64, 64) * 100
        target_zeros = torch.zeros(2, 1, 64, 64).float()

        pred_zeros = torch.ones(2, 1, 64, 64) * -100
        target_ones = torch.ones(2, 1, 64, 64).float()

        valid_mask = torch.zeros(2, 1, 64, 64)
        valid_mask[:, :, :32, :] = 1.0

        loss_smatch = loss_function(pred_ones, target_ones, valid_mask)
        loss_mismatch_1 = loss_function(pred_ones, target_zeros, valid_mask)
        loss_mismatch_2 = loss_function(pred_zeros, target_ones, valid_mask)

        assert loss_mismatch_1.item() > loss_smatch
        assert loss_mismatch_2.item() > loss_smatch

    def test_gradients_with_valid_mask(self):
        loss_function = MaskedFocalLoss(alpha=0.75, gamma=2.0)

        pred = torch.randn(2, 1, 64, 64, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        valid_mask = torch.zeros(2, 1, 64, 64)
        valid_mask[:, :, :32, :] = 1.0

        loss = loss_function(pred, target, valid_mask)
        loss.backward()

        assert pred.grad is not None
        assert not torch.all(pred.grad == 0)
        assert pred.grad.shape == pred.shape

    @pytest.mark.parametrize("batch_size", list(range(1, 33)))
    def test_different_batch_sizes_with_valid_mask(self, batch_size):
        loss_function = MaskedFocalLoss(alpha=0.75, gamma=2.0)

        pred = torch.randn(batch_size, 1, 64, 64)
        target = torch.randint(0, 2, (batch_size, 1, 64, 64)).float()
        valid_mask = torch.zeros(batch_size, 1, 64, 64)
        valid_mask[:, :, :32, :] = 1.0

        loss = loss_function(pred, target, valid_mask)
        assert loss.dim() == 0
        assert loss.item() >= 0

    @pytest.mark.parametrize(
        "height, width",
        [(i, j) for i in range(32, 513, 32) for j in range(32, 513, 32)],
    )
    def test_different_resolutions_with_valid_mask(self, height, width):
        loss_function = MaskedFocalLoss(alpha=0.75, gamma=2.0)

        pred = torch.randn(2, 1, height, width)
        target = torch.randint(0, 2, (2, 1, height, width)).float()
        valid_mask = torch.zeros(2, 1, height, width)
        valid_mask[:, :, : height // 2, :] = 1.0

        loss = loss_function(pred, target, valid_mask)

        assert loss.dim() == 0
        assert loss.item() >= 0

    @pytest.mark.parametrize("alpha", np.arange(0.05, 1, 0.05))
    def test_different_alpha_with_valid_mask(self, alpha):
        loss_function = MaskedFocalLoss(alpha=alpha, gamma=2.0)

        pred_ones = torch.ones(2, 1, 64, 64) * 100
        target_ones = torch.ones(2, 1, 64, 64).float()

        pred_zeros = torch.ones(2, 1, 64, 64) * -100
        target_zeros = torch.zeros(2, 1, 64, 64).float()

        valid_mask = torch.zeros(2, 1, 64, 64)
        valid_mask[:, :, :32, :] = 1.0

        loss_one = loss_function(pred_ones, target_ones, valid_mask)
        loss_zero = loss_function(pred_zeros, target_zeros, valid_mask)

        assert loss_zero.item() >= 0
        assert loss_one.item() >= 0
        assert loss_zero.item() == pytest.approx(0.0, abs=1e-3)
        assert loss_one.item() == pytest.approx(0.0, abs=1e-3)

    def test_alpha_monotonic_with_valid_mask(self):
        loss_functions = [MaskedFocalLoss(alpha=alpha, gamma=2.0) for alpha in np.arange(0.0, 1.0, 0.05)]

        pred_uncertain = torch.zeros(2, 1, 64, 64)
        target = torch.ones(2, 1, 64, 64).float()
        valid_mask = torch.zeros(2, 1, 64, 64)
        valid_mask[:, :, :32, :] = 1.0

        losses = [lf(pred_uncertain, target, valid_mask).item() for lf in loss_functions]

        for i in range(1, len(losses)):
            assert losses[i - 1] <= losses[i] + 1e-5

    def test_gamma_monotonic_with_valid_mask(self):
        loss_functions = [MaskedFocalLoss(alpha=0.75, gamma=gamma) for gamma in np.arange(0.0, 5.0, 0.1)]

        pred = torch.ones(2, 1, 64, 64) * 2
        target = torch.ones(2, 1, 64, 64).float()
        valid_mask = torch.zeros(2, 1, 64, 64)
        valid_mask[:, :, :32, :] = 1.0

        losses = [lf(pred, target, valid_mask).item() for lf in loss_functions]

        for i in range(1, len(losses)):
            assert losses[i] <= losses[i - 1] + 1e-5

    def test_extreme_value_with_valid_mask(self):
        loss_function = MaskedFocalLoss(alpha=0.75, gamma=2.0)

        pred_big = torch.ones(2, 1, 64, 64) * 1e6
        pred_small = torch.ones(2, 1, 64, 64) * 1e-6

        target_ones = torch.ones(2, 1, 64, 64).float()
        target_zeros = torch.zeros(2, 1, 64, 64).float()

        valid_mask = torch.zeros(2, 1, 64, 64)
        valid_mask[:, :, :32, :] = 1.0

        loss_zero_1 = loss_function(pred_big, target_zeros, valid_mask)
        loss_zero_2 = loss_function(pred_small, target_ones, valid_mask)
        loss_one_1 = loss_function(pred_big, target_ones, valid_mask)
        loss_one_2 = loss_function(pred_small, target_zeros, valid_mask)

        assert torch.isfinite(loss_zero_1)
        assert torch.isfinite(loss_zero_2)
        assert torch.isfinite(loss_one_1)
        assert torch.isfinite(loss_one_2)

        assert loss_zero_1.item() >= 0
        assert loss_zero_2.item() >= 0
        assert loss_one_1.item() >= 0
        assert loss_one_2.item() >= 0

    def test_monotonic_with_valid_mask(self):
        loss_function = MaskedFocalLoss(alpha=0.75, gamma=2.0)

        preds = [torch.ones(2, 1, 64, 64) * v for v in range(-100, 101)]
        target_ones = torch.ones(2, 1, 64, 64).float()
        valid_mask = torch.zeros(2, 1, 64, 64)
        valid_mask[:, :, :32, :] = 1.0

        losses = [loss_function(p, target_ones, valid_mask).item() for p in preds]
        for i in range(1, len(losses)):
            assert losses[i] <= losses[i - 1] + 1e-5

    def test_reproducibility_with_valid_mask(self):
        loss_function = MaskedFocalLoss(alpha=0.75, gamma=2.0)

        torch.manual_seed(42)
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        valid_mask = torch.zeros(2, 1, 64, 64)
        valid_mask[:, :, :32, :] = 1.0

        loss1 = loss_function(pred, target, valid_mask)
        loss2 = loss_function(pred, target, valid_mask)

        assert loss1.item() == pytest.approx(loss2.item(), abs=1e-10)

    def test_compared_to_bce_with_valid_mask(self):
        focal_loss_function = MaskedFocalLoss(alpha=0.5, gamma=2.0)
        bce_loss_function = torch.nn.BCEWithLogitsLoss(reduction="none")

        preds = [torch.ones(2, 1, 64, 64) * v for v in range(-100, 101)]
        target_zeros = torch.zeros(2, 1, 64, 64).float()
        target_ones = torch.ones(2, 1, 64, 64).float()
        valid_mask = torch.zeros(2, 1, 64, 64)
        valid_mask[:, :, :32, :] = 1.0

        focal_loss_1 = [focal_loss_function(p, target_zeros, valid_mask).item() for p in preds]
        focal_loss_2 = [focal_loss_function(p, target_ones, valid_mask).item() for p in preds]

        bce_loss_1 = []
        bce_loss_2 = []
        for p in preds:
            bce_unreduced = bce_loss_function(p, target_zeros)
            masked_bce = (bce_unreduced * valid_mask).sum() / valid_mask.sum()
            bce_loss_1.append(masked_bce.item())

            bce_unreduced = bce_loss_function(p, target_ones)
            masked_bce = (bce_unreduced * valid_mask).sum() / valid_mask.sum()
            bce_loss_2.append(masked_bce.item())

        for fl, bce in zip(focal_loss_1, bce_loss_1):
            assert fl <= bce + 1e-5
        for fl, bce in zip(focal_loss_2, bce_loss_2):
            assert fl <= bce + 1e-5
