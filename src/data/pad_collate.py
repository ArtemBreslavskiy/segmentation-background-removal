import torch
from typing import List, Dict, Any


def pad_collate(batch: List[Dict[str, Any]], alignment: int = 32, pad_value: float = 0.0) -> Dict[str, Any]:
    images = []
    masks = []
    sources = []
    max_h, max_w = 0, 0

    for sample in batch:
        image = sample['image']
        if isinstance(image, torch.Tensor):
            image = image.float()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        _, h, w = image.shape
        max_h = max(max_h, h)
        max_w = max(max_w, w)
        images.append(image)
        masks.append(sample.get('mask'))
        sources.append(sample.get('source', 'unknown'))

    if alignment > 1:
        max_h = ((max_h + alignment - 1) // alignment) * alignment
        max_w = ((max_w + alignment - 1) // alignment) * alignment

    padded_images, padded_masks = [], []
    for image, mask in zip(images, masks):
        h, w = image.shape[-2], image.shape[-1]
        pad_h = max_h - h
        pad_w = max_w - w
        image_pad = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), value=pad_value)
        padded_images.append(image_pad)

        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.float()
            else:
                mask = torch.from_numpy(mask).float()
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            elif mask.dim() == 3 and mask.shape[0] != 1:
                mask = mask.permute(2, 0, 1)
            h, w = mask.shape[-2], mask.shape[-1]
            pad_h = max_h - h
            pad_w = max_w - w
            mask_pad = torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h), value=pad_value)
            padded_masks.append(mask_pad)
        else:
            padded_masks.append(torch.zeros((1, max_h, max_w), dtype=torch.float32))

    return {
        'image': torch.stack(padded_images, dim=0),
        'mask': torch.stack(padded_masks, dim=0),
        'source': sources
    }