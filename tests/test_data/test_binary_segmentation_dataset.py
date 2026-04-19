import albumentations as A
import pytest
import torch

from src.data.BinarySegmentationDataset import BinarySegmentationDataset


class TestBinarySegmentationDataset:

    @pytest.mark.parametrize(
        "transforms",
        [
            None,
            A.Compose([A.HorizontalFlip()]),
            {"geometric": A.HorizontalFlip(), "final": A.ToTensorV2()},
        ],
    )
    def test_init_success(self, binary_dataset_path, transforms):
        ds = BinarySegmentationDataset(binary_dataset_path, transforms=transforms)
        assert ds.path == binary_dataset_path
        assert len(ds.images) == 3
        assert len(ds.masks) == 3
        assert ds.length == 3

    def test_init_mismatched_counts(self, mismatched_binary_path):
        with pytest.raises(ValueError, match="Mismatch between number of images"):
            BinarySegmentationDataset(mismatched_binary_path)

    def test_init_nonexistent_path(self, tmp_path):
        fake_path = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError):
            BinarySegmentationDataset(fake_path)

    def test_init_empty_dirs(self, empty_binary_path):
        ds = BinarySegmentationDataset(empty_binary_path)
        assert ds.length == 0

    def test_len(self, binary_dataset):
        assert len(binary_dataset) == 3

    def test_len_empty(self, empty_binary_path):
        ds = BinarySegmentationDataset(empty_binary_path)
        assert len(ds) == 0

    @pytest.mark.parametrize("idx", [0, 1, 2])
    def test_getitem_shapes(self, binary_dataset, idx):
        img, mask = binary_dataset[idx]
        assert isinstance(img, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert img.shape == (3, 64, 64)
        assert mask.shape == (1, 64, 64)

    @pytest.mark.parametrize("idx", [0, 1, 2])
    def test_getitem_value_ranges(self, binary_dataset, idx):
        img, mask = binary_dataset[idx]
        assert img.min() >= 0.0 and img.max() <= 1.0
        assert mask.min() >= 0.0 and mask.max() <= 1.0
        assert torch.all((mask == 0) | (mask == 1))

    @pytest.mark.parametrize("idx", [0, 1, 2])
    def test_binarization(self, binary_dataset, idx):
        _, mask = binary_dataset[idx]
        unique = torch.unique(mask)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_compose_transforms(self, binary_dataset_with_compose):
        img, mask = binary_dataset_with_compose[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

    def test_dict_transforms_with_final(self, binary_dataset_with_dict):
        img, mask = binary_dataset_with_dict[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

    @pytest.mark.parametrize("key", ["geometric", "photometric"])
    def test_single_dict_transform(self, binary_dataset_path, key):
        transforms = {
            key: (
                A.HorizontalFlip()
                if key == "geometric"
                else A.RandomBrightnessContrast()
            ),
            "final": A.ToTensorV2(),
        }
        ds = BinarySegmentationDataset(binary_dataset_path, transforms=transforms)
        img, mask = ds[0]
        assert img.shape == (3, 64, 64)
        assert mask.shape == (1, 64, 64)

    def test_missing_mask_raises(self, missing_mask_path):
        ds = BinarySegmentationDataset(missing_mask_path)
        img1, mask1 = ds[0]
        assert img1.shape == (3, 64, 64)
        with pytest.raises(FileNotFoundError, match="Mask not found"):
            _ = ds[1]

    def test_missing_image_raises(self, missing_image_path):
        ds = BinarySegmentationDataset(missing_image_path)
        img1, mask1 = ds[0]
        assert img1.shape == (3, 64, 64)
        with pytest.raises(FileNotFoundError, match="Image not found"):
            _ = ds[1]

    @pytest.mark.parametrize(
        "transforms",
        [
            A.Compose(
                [A.RandomRotate90(), A.GaussianBlur(blur_limit=3), A.ToTensorV2()]
            ),
            {
                "geometric": A.Compose([A.RandomRotate90()]),
                "final": A.Compose([A.ToTensorV2()]),
            },
        ],
    )
    def test_binarization_after_transforms(self, binary_dataset_path, transforms):
        ds = BinarySegmentationDataset(binary_dataset_path, transforms=transforms)
        _, mask = ds[0]
        unique = torch.unique(mask)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_dataloader(self, binary_dataset, batch_size):
        loader = torch.utils.data.DataLoader(binary_dataset, batch_size=batch_size)
        for images, masks in loader:
            expected_batch_size = min(batch_size, len(binary_dataset))
            assert images.shape[0] == expected_batch_size
            assert images.shape[1:] == (3, 64, 64)
            assert masks.shape[1:] == (1, 64, 64)
            break

    def test_dataloader_with_transforms(self, binary_dataset_with_compose):
        loader = torch.utils.data.DataLoader(binary_dataset_with_compose, batch_size=2)
        for images, masks in loader:
            assert images.shape[0] == 2
            assert images.shape[1:] == (3, 64, 64)
            assert masks.shape[1:] == (1, 64, 64)
            break

    @pytest.mark.parametrize("idx", range(3))
    def test_dtypes(self, binary_dataset, idx):
        img, mask = binary_dataset[idx]
        assert img.dtype == torch.float32
        assert mask.dtype == torch.float32

    @pytest.mark.parametrize(
        "transforms_type",
        [
            None,
            A.Compose([A.HorizontalFlip(), A.ToTensorV2()]),
            {"geometric": A.HorizontalFlip(), "final": A.ToTensorV2()},
            {"photometric": A.RandomBrightnessContrast(), "final": A.ToTensorV2()},
            {"final": A.ToTensorV2()},
            {"geometric": A.HorizontalFlip(), "final": A.ToTensorV2()},
            {
                "geometric": A.HorizontalFlip(),
                "photometric": A.RandomBrightnessContrast(),
                "final": A.ToTensorV2(),
            },
        ],
    )
    def test_various_transforms(self, binary_dataset_path, transforms_type):
        ds = BinarySegmentationDataset(binary_dataset_path, transforms=transforms_type)
        img, mask = ds[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
