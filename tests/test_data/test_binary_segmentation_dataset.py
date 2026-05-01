import pytest
import torch
import numpy as np
import cv2
import json
from pathlib import Path

from src.data.BinarySegmentationDataset import BinarySegmentationDataset


class TestBinarySegmentationDataset:
    def test_init_with_manifest(self, dummy_manifest):
        dataset = BinarySegmentationDataset(manifest=dummy_manifest)
        assert len(dataset) == 3
        assert dataset.images[0] == Path(dummy_manifest[0]["image"])
        assert dataset.images[1] == Path(dummy_manifest[1]["image"])
        assert dataset.images[2] == Path(dummy_manifest[2]["image"])
        assert dataset.masks[0] == Path(dummy_manifest[0]["mask"])
        assert dataset.masks[1] == Path(dummy_manifest[1]["mask"])
        assert dataset.masks[2] == Path(dummy_manifest[2]["mask"])

    def test_init_with_json_path(self, tmp_path, dummy_manifest):
        json_path = tmp_path / "manifest"
        with open(json_path, 'w') as f:
            json.dump(dummy_manifest, f)

        dataset = BinarySegmentationDataset(json_path=json_path)
        assert len(dataset) == 3
        assert dataset.images[0] == Path(dummy_manifest[0]["image"])
        assert dataset.images[1] == Path(dummy_manifest[1]["image"])
        assert dataset.images[2] == Path(dummy_manifest[2]["image"])
        assert dataset.masks[0] == Path(dummy_manifest[0]["mask"])
        assert dataset.masks[1] == Path(dummy_manifest[1]["mask"])
        assert dataset.masks[2] == Path(dummy_manifest[2]["mask"])

    def test_no_manifest(self):
        with pytest.raises(ValueError, match="Either json_path or manifest must be provided"):
            dataset = BinarySegmentationDataset()

    def test_empty_manifest(self):
        with pytest.raises(ValueError, match="Either json_path or manifest must be provided"):
            dataset = BinarySegmentationDataset(manifest=[])

    def test_invalid_resize_mode(self, dummy_manifest):
        with pytest.raises(ValueError, match="Unknown mode"):
            dataset = BinarySegmentationDataset(manifest=dummy_manifest, resize_mode="invalid")

    def test_getitem(self, dummy_manifest):
        dataset = BinarySegmentationDataset(manifest=dummy_manifest)
        for data in dataset:
            img, mask = data
            assert len(data) == 2

            assert isinstance(img, torch.Tensor)
            assert img.dtype == torch.float32
            assert img.ndim == 3
            assert img.shape[0] == 3

            assert isinstance(mask, torch.Tensor)
            assert mask.dtype == torch.float32
            assert mask.ndim == 3
            assert mask.shape[0] == 1
            assert torch.all((mask == 0) | (mask == 1))

    def test_getitem_no_image(self, dummy_manifest):
        dummy_manifest[0]["image"] = "/nonexistent/path.png"
        dataset = BinarySegmentationDataset(manifest=dummy_manifest)
        with pytest.raises(FileNotFoundError, match="Image not found"):
            img, mask = dataset[0]

    def test_getitem_no_mask(self, dummy_manifest):
        dummy_manifest[0]["mask"] = "/nonexistent/path.png"
        dataset = BinarySegmentationDataset(manifest=dummy_manifest)
        with pytest.raises(FileNotFoundError, match="Mask not found"):
            img, mask = dataset[0]

    @pytest.mark.parametrize("resize_mode", ["resize", "crop"])
    def test_getitem_max_resize_square(self, dummy_manifest, resize_mode):
        dataset = BinarySegmentationDataset(manifest=dummy_manifest, max_area=32*32)
        for data in dataset:
            img, mask = data
            assert img.shape[1] == pytest.approx(32, abs=5)
            assert img.shape[2] == pytest.approx(32, abs=5)
            assert mask.shape[1] == pytest.approx(32, abs=5)
            assert mask.shape[2] == pytest.approx(32, abs=5)

    @pytest.mark.parametrize("resize_mode", ["resize", "crop"])
    def test_getitem_max_resize_rectangle(self, tmp_path, resize_mode):
        h, w = 32, 64
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[4:12, 20:40] = 255

        img_path = tmp_path / "img.png"
        mask_path = tmp_path / "mask.png"
        cv2.imwrite(str(img_path), img)
        cv2.imwrite(str(mask_path), mask)
        manifest = [{"image": str(img_path), "mask": str(mask_path),
                     "resolution": [32, 64], "source": "test"}]

        dataset = BinarySegmentationDataset(manifest=manifest, max_area=32*32, resize_mode=resize_mode)
        img, mask = dataset[0]
        new_h, new_w = img.shape[1], img.shape[2]

        assert dataset.resize_mode == resize_mode
        assert new_h * new_w <= 32*32
        assert new_h / new_w == pytest.approx(h / w, abs=0.1)

    def test_basic_transforms_applied(self, dummy_manifest, dummy_basic_transforms):
        dataset = BinarySegmentationDataset(manifest=dummy_manifest, transforms=dummy_basic_transforms)
        for data in dataset:
            img, mask = data
            assert isinstance(img, torch.Tensor)
            assert isinstance(mask, torch.Tensor)
            assert img.shape[0] == 3
            assert mask.shape[0] == 1
            assert img.min() < 0

    def test_dict_transforms_applied(self, dummy_manifest, dummy_dict_transforms):
        dataset = BinarySegmentationDataset(manifest=dummy_manifest, transforms=dummy_dict_transforms)
        for data in dataset:
            img, mask = dataset[0]
            assert isinstance(img, torch.Tensor)
            assert isinstance(mask, torch.Tensor)
            assert img.shape[0] == 3
            assert mask.shape[0] == 1
            assert img.min() < 0