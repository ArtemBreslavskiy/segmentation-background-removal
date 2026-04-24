from pathlib import Path
import json
from typing import Dict, Optional, Union, List

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
    ):
        self.path = Path(json_path)
        self.transforms = transforms
        if not manifest:
            if json_path:
                with open(json_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
            else:
                raise ValueError("Either json_path or manifest must be provided")
        self.images = [Path(item['image']) for item in manifest]
        self.masks = [Path(item['mask']) for item in manifest]
        self.sources = [item.get('source', 'unknown') for item in manifest]
        self.length = len(self.images)

    def __getitem__(self, idx):
        path_image = self.images[idx]
        path_mask = self.masks[idx]
        source = self.sources[idx]

        image = cv2.imread(str(path_image))
        if image is None:
            raise FileNotFoundError(f"Image not found: {path_image}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0

        mask = cv2.imread(str(path_mask), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {path_mask}")
        mask = (mask != 0).astype(np.float32)

        if self.transforms:
            if isinstance(self.transforms, dict):
                if "geometric" in self.transforms:
                    transformed = self.transforms["geometric"](image=image, mask=mask)
                    image = transformed["image"]
                    mask = transformed["mask"]

                if (
                    "photometric" in self.transforms
                    and self.transforms["photometric"] is not None
                ):
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

        else:
            image = torch.from_numpy(image).float().permute(2, 0, 1)
            mask = torch.from_numpy(mask).float().unsqueeze(0)

        mask = (mask > 0.5).float()

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        return {"image": image, "mask": mask, "source": source}

    def __len__(self):
        return self.length
