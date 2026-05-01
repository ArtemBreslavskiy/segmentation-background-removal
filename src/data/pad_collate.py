from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def pad_collate(
    batch: List[Tuple[Union[torch.Tensor, ]]],
    alignment: int = 32,
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not batch:
        raise ValueError("Butch cannot be empty")

    max_h, max_w = 0, 0
    images = []
    masks = []
    valid_masks = []

    for sample in batch:
        image, mask = sample[0], sample[1]
        if isinstance(image, torch.Tensor):
            image = image.float()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        if isinstance(mask, torch.Tensor):
            mask = mask.float()
        else:
            mask = torch.from_numpy(mask).float()
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        h, w = image.shape[-2], image.shape[-1]
        max_h = max(max_h, h)
        max_w = max(max_w, w)
        valid = torch.ones(1, h, w, dtype=torch.float32)
        images.append((image, h, w))
        masks.append((mask, h, w))
        valid_masks.append((valid, h, w))

    if alignment > 1:
        max_h = ((max_h + alignment - 1) // alignment) * alignment
        max_w = ((max_w + alignment - 1) // alignment) * alignment

    padded_images, padded_masks, padded_valid = [], [], []
    for img_tuple, mask_tuple, valid_tuple in zip(images, masks, valid_masks):
        image, h, w = img_tuple
        mask = mask_tuple[0]
        valid = valid_tuple[0]
        pad_h = max_h - h
        pad_w = max_w - w
        image_pad = F.pad(image, (0, pad_w, 0, pad_h), value=pad_value)
        mask_pad = F.pad(mask, (0, pad_w, 0, pad_h), value=pad_value)
        valid_pad = F.pad(valid, (0, pad_w, 0, pad_h), value=0.0)
        padded_images.append(image_pad)
        padded_masks.append(mask_pad)
        padded_valid.append(valid_pad)

    return (torch.stack(padded_images, dim=0), torch.stack(padded_masks, dim=0), torch.stack(padded_valid, dim=0))
