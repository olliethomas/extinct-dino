from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from torch import Tensor

__all__ = ["DINOAugmentation"]


class DINOAugmentation(A.ImageOnlyTransform):
    def __init__(
        self,
        global_crops_scale: tuple[float, float],
        local_crops_scale: tuple[float, float],
        local_crops_number: int,
    ) -> None:
        super().__init__()
        flip_and_color_jitter = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                A.ToGray(p=0.2),
            ]
        )
        normalize = A.Compose(
            [
                ToTensorV2(),
                # The default per-mean/std values for albumentations.Normalize
                # match the ImageNet ones prescribed by DINO
                A.Normalize(),
                A.ToFloat(),
            ]
        )
        self.global_transfo1 = A.Compose(
            [
                A.RandomResizedCrop(
                    height=224, width=224, scale=global_crops_scale, interpolation=cv2.INTER_CUBIC  # type: ignore
                ),
                flip_and_color_jitter,
                A.GaussianBlur(p=1.0, sigma_limit=(0.1, 2)),
                *normalize,
            ]
        )
        self.global_transfo2 = A.Compose(
            [
                A.RandomResizedCrop(
                    height=224, width=224, scale=global_crops_scale, interpolation=cv2.INTER_CUBIC  # type: ignore
                ),
                flip_and_color_jitter,
                A.GaussianBlur(p=1.0, sigma_limit=(0.1, 2)),
                A.Solarize(p=0.2),
                *normalize,
            ]
        )
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = A.Compose(
            [
                A.RandomResizedCrop(
                    height=96, width=96, scale=local_crops_scale, interpolation=cv2.INTER_CUBIC  # type: ignore
                ),
                flip_and_color_jitter,
                A.GaussianBlur(p=1.0, sigma_limit=(0.1, 2)),
                *normalize,
            ]
        )

    def __call__(self, image: np.ndarray, **kwargs: Any) -> list[Tensor]:
        crops = []
        crops.append(self.global_transfo1(image=image)["image"])
        crops.append(self.global_transfo2(image=image)["image"])
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image=image)["image"])
        return crops
