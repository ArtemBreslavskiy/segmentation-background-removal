import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import albumentations as A
import cv2
import numpy as np
import torch
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2


class BinarySegmentationDataset(data.Dataset):
    def __init__(
        self,
        json_path: Optional[Union[str, Path]] = None,
        manifest: Optional[List[Dict]] = None,
        transforms: Optional[Union[A.Compose, Dict[str, A.Compose]]] = None,
        max_area: int = 0,
        resize_mode: str = "resize",
        area_threshold_mix: int = 0,
        min_foreground_share: int = 0,
    ):
        resize_mode = resize_mode.lower()
        resize_modes = ["resize", "crop", "mix-a", "mix-b"]
        if resize_mode not in resize_modes:
            raise ValueError(f"Unknown mode: {resize_mode}. Available mods: {resize_modes}")
        if resize_mode in ["mix-a", "mix-b"] and area_threshold_mix < max_area:
            raise ValueError("When resize_mode is mix-a or mix-b, area_threshold_mix cannot be less than max_area")
        if not manifest:
            if json_path:
                with open(json_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            else:
                raise ValueError("Either json_path or manifest must be provided")

        self.manifest = manifest
        self.transforms = transforms if transforms else ToTensorV2()
        self.images = [Path(item["image"]) for item in manifest]
        self.masks = [Path(item["mask"]) for item in manifest]
        self.max_area = max_area
        self.resize_mode = resize_mode
        self.area_threshold_mixed = area_threshold_mix
        self.min_foreground_share = min_foreground_share
        self.length = len(self.images)

    def _resize(self, image, mask, h, w):
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return image, mask

    def _crop(self, image, mask, h, w, min_foreground=50):
        for _ in range(10):
            crop = A.RandomCrop(height=h, width=w, p=1.0)
            transformed = crop(image=image, mask=mask)
            candidate_mask = transformed["mask"]
            if candidate_mask.sum() >= min_foreground:
                return transformed["image"], candidate_mask
        crop = A.CenterCrop(height=h, width=w, p=1.0)
        transformed = crop(image=image, mask=mask)
        return transformed["image"], transformed["mask"]

    def _get_resized(self, image, mask):
        h, w = image.shape[:2]
        area = h * w
        if area > self.max_area:
            scale = np.sqrt(self.max_area / (h * w))
            new_h = int(h * scale)
            new_w = int(w * scale)

            if self.resize_mode == "resize":
                return self._resize(image, mask, new_h, new_w)
            if self.resize_mode == "crop":
                return self._crop(image, mask, new_h, new_w, int(new_h * new_w * self.min_foreground_share))

            if self.resize_mode == "mix-a":
                if area > self.area_threshold_mixed:
                    return self._resize(image, mask, new_h, new_w)
                else:
                    return self._crop(image, mask, new_h, new_w, int(new_h * new_w * self.min_foreground_share))

            if self.resize_mode == "mix-b":
                if area > self.area_threshold_mixed:
                    scale = np.sqrt(self.area_threshold_mixed / (h * w))
                    intermediate_h = int(h * scale)
                    intermediate_w = int(w * scale)
                    image, mask = self._resize(image, mask, intermediate_h, intermediate_w)
                    return self._crop(image, mask, new_h, new_w, int(new_h * new_w * self.min_foreground_share))
                else:
                    return self._crop(image, mask, new_h, new_w, int(new_h * new_w * self.min_foreground_share))
        return image, mask

    def get_manifest_with_correct_resolution(self):
        if self.max_area < 1:
            raise ValueError("max_area cannot be less than 1 when using get_manifest_with_correct_resolution")
        for i in range(len(self.manifest)):
            h, w = self.manifest[i]["resolution"]
            if h * w > self.max_area:
                scale = np.sqrt(self.max_area / (h * w))
                new_h = int(h * scale)
                new_w = int(w * scale)
                self.manifest[i]["resolution"] = [new_h, new_w]
        return self.manifest

    def __getitem__(self, idx):
        path_image = self.images[idx]
        path_mask = self.masks[idx]

        image = cv2.imread(str(path_image))
        mask = cv2.imread(str(path_mask), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {path_image}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {path_mask}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.max_area >= 1:
            image, mask = self._get_resized(image, mask)

        image = image.astype("float32") / 255.0
        mask = (mask != 0).astype(np.float32)

        if self.transforms:
            if isinstance(self.transforms, dict):
                if "geometric" in self.transforms and self.transforms["geometric"] is not None:
                    transformed = self.transforms["geometric"](image=image, mask=mask)
                    image = transformed["image"]
                    mask = transformed["mask"]

                if "photometric" in self.transforms and self.transforms["photometric"] is not None:
                    image = self.transforms["photometric"](image=image)["image"]

                if "final_image" in self.transforms:
                    image = self.transforms["final_image"](image=image)["image"]
                elif "final" in self.transforms:
                    aug_final = self.transforms["final"](image=image, mask=mask)
                    image = aug_final["image"]
                    mask = aug_final["mask"]

                if "final_mask" in self.transforms:
                    mask = self.transforms["final_mask"](image=mask)["image"]
                elif "final" not in self.transforms:
                    mask = torch.from_numpy(mask).float().unsqueeze(0)

            else:
                augmented = self.transforms(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

        mask = (mask > 0.5).float()

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        if isinstance(image, torch.Tensor) and image.shape[0] != 3:
            image = image.expand(3, -1, -1)

        return image, mask

    def __len__(self):
        return self.length
