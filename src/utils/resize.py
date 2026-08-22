import cv2
import albumentations as A
import numpy as np


def resize(image, mask, new_h: int, new_w: int):
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return image, mask


def crop(image, mask, new_h: int, new_w: int, min_foreground: int = 50):
    if new_h > image.shape[0] or new_w > image.shape[1]:
        raise ValueError(f"Crop size ({new_h},{new_w}) exceeds image size {image.shape[:2]}")

    for _ in range(10):
        crop = A.RandomCrop(height=new_h, width=new_w, p=1.0)
        transformed = crop(image=image, mask=mask)
        candidate_mask = transformed["mask"]
        if candidate_mask.sum() >= min_foreground:
            return transformed["image"], candidate_mask

    return resize(image, mask, new_h, new_w)


def resize_mix_a(image, mask, new_area: int, area_threshold_mixed: int, min_foreground: int = 50):
    h, w = image.shape[:2]
    area = h * w
    new_h, new_w = get_scale_hw(h=h, w=w, new_area=new_area)

    if area > area_threshold_mixed:
        return resize(image, mask, new_h=new_h, new_w=new_w)
    else:
        if new_h <= image.shape[0] and new_w <= image.shape[1]:
            return crop(image, mask, new_h, new_w, min_foreground=min_foreground)
        else:
            return resize(image, mask, new_h, new_w)


def resize_mix_b(image, mask, new_area: int, area_threshold_mixed: int, min_foreground: int = 50):
    h, w = image.shape[:2]
    area = h * w
    new_h, new_w = get_scale_hw(h=h, w=w, new_area=new_area)

    if area > area_threshold_mixed:
        intermediate_h, intermediate_w = get_scale_hw(h=h, w=w, new_area=area_threshold_mixed)
        image, mask = resize(image, mask, new_h=intermediate_h, new_w=intermediate_w)
        if new_h <= image.shape[0] and new_w <= image.shape[1]:
            return crop(image, mask, new_h, new_w, min_foreground=min_foreground)
        else:
            return resize(image, mask, new_h, new_w)
    else:
        if new_h <= image.shape[0] and new_w <= image.shape[1]:
            return crop(image, mask, new_h, new_w, min_foreground=min_foreground)
        else:
            return resize(image, mask, new_h, new_w)


def get_scale_hw(h: int, w: int, new_area: int) -> tuple[int, int]:
    scale = np.sqrt(new_area / (h * w))
    new_h = int(h * scale)
    new_w = int(w * scale)
    return new_h, new_w
