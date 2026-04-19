from typing import Dict, Union

import albumentations as A
import albumentations.pytorch as AP
import cv2


def get_train_transforms(
    h: int = 384, w: int = 384
) -> Dict[str, Union[A.Compose, A.BasicTransform]]:
    geometric = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.1, 0.1),
                rotate=(-20, 20),
                shear=(-5, 5),
                p=0.7,
                border_mode=cv2.BORDER_REFLECT_101,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
            ),
            A.Resize(
                h,
                w,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=1.0,
            ),
        ],
        additional_targets={"mask": "image"},
    )

    photometric = A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5
            ),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, int(h * 0.15)),
                hole_width_range=(8, int(w * 0.15)),
                fill=0,
                p=0.25,
            ),
            A.GaussNoise(std_range=(0.05, 0.15), p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.2,
            ),
        ]
    )

    final_image = A.Compose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            AP.ToTensorV2(),
        ]
    )

    final_mask = AP.ToTensorV2()

    return {
        "geometric": geometric,
        "photometric": photometric,
        "final_image": final_image,
        "final_mask": final_mask,
    }


def get_val_test_transforms(
    h: int = 384, w: int = 384
) -> Dict[str, Union[A.Compose, A.BasicTransform]]:
    geometric = A.Compose(
        [
            A.Resize(
                h,
                w,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=1.0,
            ),
        ],
        additional_targets={"mask": "image"},
    )

    final_image = A.Compose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            AP.ToTensorV2(),
        ]
    )

    final_mask = AP.ToTensorV2()

    return {
        "geometric": geometric,
        "photometric": None,
        "final_image": final_image,
        "final_mask": final_mask,
    }
