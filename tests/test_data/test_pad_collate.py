import pytest
from typing import Callable

from src.data.pad_collate import pad_collate


class TestPadCollate:
    def test_init(self):
        fn = pad_collate
        assert isinstance(pad_collate, Callable)

    def test_init_without_batch(self):
        with pytest.raises(ValueError, match="Butch cannot be empty"):
            pad_collate(batch=None)

    def test_init_with_empty_batch(self):
        with pytest.raises(ValueError, match="Butch cannot be empty"):
            pad_collate(batch=[])

    def test_padding(self, batch):
        imgs, masks, valids = pad_collate(batch)
        padded_img1, padded_img2 = imgs[0], imgs[1]
        padded_mask1, padded_mask2 = masks[0], masks[1]
        valid1, valid2 = valids[0], valids[1]

        assert (padded_img1.shape[1] == padded_img2.shape[1] == padded_mask1.shape[1]
                == padded_mask2.shape[1] == valid1.shape[1] == valid2.shape[1])
        assert (padded_img1.shape[2] == padded_img2.shape[2] == padded_mask1.shape[2]
                == padded_mask2.shape[2] == valid1.shape[2] == valid2.shape[2])

    def test_valid_mask(self, batch):
        data1, data2 = batch
        img1, mask1 = data1
        img2, mask2 = data2

        _, _, valids = pad_collate(batch)
        valid1, valid2 = valids[0], valids[1]

        assert valid1.sum() == 4096
        assert valid2.sum() == 1024

    def test_pad_value(self, batch):
        imgs, _, _ = pad_collate(batch, pad_value=0.5)
        padded_img1, padded_img2 = imgs[0], imgs[1]
        assert padded_img2[0, -1, -1] == 0.5

    def test_alignment(self, batch):
        imgs, masks, valids = pad_collate(batch, alignment=13)
        padded_img1, padded_img2 = imgs[0], imgs[1]
        padded_mask1, padded_mask2 = masks[0], masks[1]
        valid1, valid2 = valids[0], valids[1]

        assert padded_img1.shape[1] % 13 == 0
        assert padded_img1.shape[2] % 13 == 0
        assert padded_img2.shape[1] % 13 == 0
        assert padded_img2.shape[2] % 13 == 0

        assert padded_mask1.shape[1] % 13 == 0
        assert padded_mask1.shape[2] % 13 == 0
        assert padded_mask2.shape[1] % 13 == 0
        assert padded_mask2.shape[2] % 13 == 0

        assert valid1.shape[1] % 13 == 0
        assert valid1.shape[2] % 13 == 0
        assert valid2.shape[1] % 13== 0
        assert valid2.shape[2] % 13 == 0