import json
import numpy as np
import torch.utils.data as data
import cv2
from pathlib import Path
from src.utils.resize import resize, crop, resize_mix_a, resize_mix_b, get_scale_hw
from src.data.transforms import Transforms


class BinarySegmentationDataset(data.Dataset):
    AVAILABLE_RESIZE_MODES = ["resize", "crop", "mix-a", "mix-b"]

    def __init__(
        self,
        json_path: str | Path | None = None,
        manifest: list[dict] | None = None,
        transforms: Transforms | None = None,
        resize_mode: str = "resize",
        max_area: int = 512 * 512,
        area_threshold_mix: int = 1024 * 1024,
        min_foreground_share: float = 0,
    ):
        resize_mode = resize_mode.lower()
        self._validate_resize_mode(resize_mode)
        self._validate_max_area(max_area)
        self._validate_area_threshold_mix(area_threshold_mix, max_area)
        self._validate_min_foreground_share(min_foreground_share)

        if not manifest:
            if json_path:
                with open(json_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            else:
                raise ValueError("Either json_path or manifest must be provided")

        self.manifest = manifest
        self.transforms = transforms
        self.images = [Path(item["image"]) for item in manifest]
        self.masks = [Path(item["mask"]) for item in manifest]
        self.resize_mode = resize_mode
        self.max_area = max_area
        self.area_threshold_mixed = area_threshold_mix
        self.min_foreground_share = min_foreground_share
        self.length = len(self.images)

    @staticmethod
    def _validate_resize_mode(resize_mode: str) -> None:
        if resize_mode not in BinarySegmentationDataset.AVAILABLE_RESIZE_MODES:
            raise ValueError(
                f"Unknown mode: {resize_mode}. "
                f"Available mods: {BinarySegmentationDataset.AVAILABLE_RESIZE_MODES}"
            )

    @staticmethod
    def _validate_max_area(max_area) -> None:
        if max_area < 0:
            raise ValueError("max_area cannot be less than 0")

    @staticmethod
    def _validate_area_threshold_mix(area_threshold_mix, max_area) -> None:
        if area_threshold_mix < 1:
            raise ValueError("area_threshold_mix cannot be less than 1")
        if area_threshold_mix < max_area:
            raise ValueError("")

    @staticmethod
    def _validate_min_foreground_share(min_foreground_share) -> None:
        if min_foreground_share < 0:
            raise ValueError("min_foreground_share cannot be less than 0")
        if min_foreground_share > 1:
            raise ValueError("")

    def _resize(self, image, mask):
        if self.resize_mode == "resize":
            h, w = image.shape[:2]
            new_h, new_w = get_scale_hw(h, w, self.max_area)
            return resize(image, mask, new_h, new_w)

        if self.resize_mode == "crop":
            h, w = image.shape[:2]
            new_h, new_w = get_scale_hw(h, w, self.max_area)
            min_foreground = int(new_h * new_w * self.min_foreground_share)
            return crop(image, mask, new_h, new_w, min_foreground=min_foreground)

        if self.resize_mode == "mix-a":
            min_foreground = int(self.max_area * self.min_foreground_share)
            return resize_mix_a(
                image=image,
                mask=mask,
                new_area=self.max_area,
                area_threshold_mixed=self.area_threshold_mixed,
                min_foreground=min_foreground,
            )

        if self.resize_mode == "mix-b":
            min_foreground = int(self.max_area * self.min_foreground_share)
            return resize_mix_b(
                image=image,
                mask=mask,
                new_area=self.max_area,
                area_threshold_mixed=self.area_threshold_mixed,
                min_foreground=min_foreground,
            )

    def get_manifest_with_correct_resolution(self):
        if self.max_area > 0:
            for i, item in enumerate(self.manifest):
                h, w = item["resolution"]
                if h * w > self.max_area:
                    new_h, new_w = get_scale_hw(h, w, self.max_area)
                    item["resolution"] = [new_h, new_w]
        return self.manifest

    def __getitem__(self, idx):
        path_image = self.images[idx]
        path_mask = self.masks[idx]

        image = cv2.imread(str(path_image))
        if image is None:
            raise FileNotFoundError(f"Image not found: {path_image}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(path_mask), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {path_mask}")

        if self.max_area >= 1:
            image, mask = self._resize(image, mask)
        mask = (mask != 0).astype(np.float32)

        if self.transforms:
            if self.transforms.geometric:
                transformed = self.transforms.geometric(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
            mask = (mask > 0.5).astype(np.float32)

            if self.transforms.photometric:
                image = self.transforms.photometric(image=image)["image"]

            if self.transforms.final_image:
                image = self.transforms.final_image(image=image)["image"]

            if self.transforms.final_mask:
                mask = self.transforms.final_mask(image=mask)["image"]

        return image, mask

    def __len__(self) -> int:
        return self.length
