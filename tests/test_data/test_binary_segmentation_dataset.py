import pytest
import torch
import numpy as np
import cv2
import json
from pathlib import Path

from src.data.BinarySegmentationDataset import BinarySegmentationDataset


class TestBinarySegmentationDataset:
    def test_init_with_manifest(self, manifest):
        dataset = BinarySegmentationDataset(manifest=manifest)
        assert len(dataset) == 1
        assert dataset.images[0] == Path(manifest[0]["image"])

    def test_init_with_json_path(self, tmp_path, manifest):
        json_path = tmp_path / "manifest"
        with open(json_path, 'w') as f:
            json.dump(manifest, f)

        dataset = BinarySegmentationDataset(json_path=json_path)
        assert len(dataset) == 1
        assert dataset.images[0] == Path(manifest[0]["image"])

    def test_no_manifest(self):
        with pytest.raises(ValueError, match="Either json_path or manifest must be provided"):
            dataset = BinarySegmentationDataset()

    def test_empty_manifest(self):
        with pytest.raises(ValueError, match="Either json_path or manifest must be provided"):
            dataset = BinarySegmentationDataset(manifest=[])

    def test_invalid_resize_mode(self, manifest):
        with pytest.raises(ValueError, match="Unknown mode"):
            dataset = BinarySegmentationDataset(manifest=manifest, resize_mode="invalid")

    def test_getitem(self, manifest):
        dataset = BinarySegmentationDataset(manifest=manifest)
        data = (dataset[0])
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

    def test_getitem_no_image(self, manifest):
        manifest[0]["image"] = "/nonexistent/path.png"
        dataset = BinarySegmentationDataset(manifest=manifest)
        with pytest.raises(FileNotFoundError, match="Image not found"):
            img, mask = dataset[0]

    def test_getitem_no_mask(self, manifest):
        manifest[0]["mask"] = "/nonexistent/path.png"
        dataset = BinarySegmentationDataset(manifest=manifest)
        with pytest.raises(FileNotFoundError, match="Mask not found"):
            img, mask = dataset[0]

    def test_getitem_max_resize_square(self, manifest):
        dataset = BinarySegmentationDataset(manifest=manifest, max_area=32*32)
        img, mask = dataset[0]

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

    def test_basic_transforms_applied(self, manifest, basic_transforms):
        dataset = BinarySegmentationDataset(manifest=manifest, transforms=basic_transforms)
        img, mask = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert img.shape[0] == 3
        assert mask.shape[0] == 1
        assert img.min() < 0

    def test_dict_transforms_applied(self, manifest, dict_transforms):
        dataset = BinarySegmentationDataset(manifest=manifest, transforms=dict_transforms)
        img, mask = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert img.shape[0] == 3
        assert mask.shape[0] == 1
        assert img.min() < 0