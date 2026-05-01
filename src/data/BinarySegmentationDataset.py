import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import albumentations as A
import cv2
import numpy as np
import torch
import torch.utils.data as data


class BinarySegmentationDataset(data.Dataset):
    def __init__(
        self,
        json_path: Optional[Union[str, Path]] = None,
        manifest: Optional[List[Dict]] = None,
        transforms: Optional[Union[A.Compose, Dict[str, A.Compose]]] = None,
        max_area: int = 0,
        resize_mode: str = "resize",
    ):
        resize_mode = resize_mode.lower()
        resize_modes = ["resize", "crop"]
        if resize_mode not in resize_modes:
            raise ValueError(f"Unknown mode: {resize_mode}. Available mods: {resize_modes}")
        if not manifest:
            if json_path:
                with open(json_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            else:
                raise ValueError("Either json_path or manifest must be provided")

        self.transforms = transforms
        self.images = [Path(item["image"]) for item in manifest]
        self.masks = [Path(item["mask"]) for item in manifest]
        self.max_area = max_area
        self.resize_mode = resize_mode
        if self.max_area > 0:
            self.areas = [item.get("resolution", [0, 0])[0] * item.get("resolution", [0, 0])[1] for item in manifest]
        self.length = len(self.images)

    def __getitem__(self, idx):
        path_image = self.images[idx]
        path_mask = self.masks[idx]
        if self.max_area > 0:
            area = self.areas[idx]

        image = cv2.imread(str(path_image))
        mask = cv2.imread(str(path_mask), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {path_image}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {path_mask}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.max_area > 0:
            if area > self.max_area:
                h, w = image.shape[:2]
                scale = np.sqrt(self.max_area / (h * w))
                new_h = int(h * scale)
                new_w = int(w * scale)
                if self.resize_mode == "resize":
                    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                elif self.resize_mode == "crop":
                    crop = A.RandomCrop(height=new_h, width=new_w, p=1.0)
                    transformed = crop(image=image, mask=mask)
                    image = transformed["image"]
                    mask = transformed["mask"]

        image = image.astype("float32") / 255.0
        mask = (mask != 0).astype(np.float32)

        if self.transforms:
            if isinstance(self.transforms, dict):
                if "geometric" in self.transforms:
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

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float().permute(2, 0, 1)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float().unsqueeze(0)

        mask = (mask > 0.5).float()

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        return image, mask

    def __len__(self):
        return self.length
