"""CelebA DataModule."""
from __future__ import annotations
from typing import Any, ClassVar, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
import ethicml as em
from ethicml import implements
import ethicml.vision as emvi
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data.dataset import random_split
from torchvision import transforms as TF

from extinct.datamodules.base import VisionDataModule
from extinct.datamodules.structures import TiWrapper, AlbumentationsDataset

__all__ = ["CelebaDataModule"]


class CelebaDataModule(VisionDataModule):
    """CelebA Dataset."""

    num_classes: ClassVar[int] = 2
    num_sens: ClassVar[int] = 2

    def __init__(
        self,
        data_dir: Optional[str] = None,
        image_size: int = 64,
        batch_size: int = 32,
        num_workers: int = 0,
        val_split: float = 0.2,
        test_split: float = 0.2,
        y_label: str = "Smiling",
        s_label: str = "Male",
        seed: int = 0,
        persist_workers: bool = False,
        stratified_sampling: bool = False,
        sample_with_replacement: bool = True,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            s_dim=1,
            y_dim=1,
            seed=seed,
            persist_workers=persist_workers,
            stratified_sampling=stratified_sampling,
            sample_with_replacement=sample_with_replacement,
        )
        self.image_size = image_size
        self.dims = (3, self.image_size, self.image_size)
        self.y_label = y_label
        self.s_label = s_label

    @implements(LightningDataModule)
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        _, _ = em.celeba(
            download_dir=self.data_dir,
            label=self.y_label,
            sens_attr=self.s_label,
            download=True,
            check_integrity=True,
        )

    @property
    def _base_augmentations(self) -> list[A.BasicTransform]:
        tform_ls = [
            A.Resize(self.image_size, self.image_size),
            A.CenterCrop(self.image_size, self.image_size),
        ]
        return tform_ls

    def _train_augmentations(self) -> list[A.BasicTransform]:
        tform_ls = [
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
        return tform_ls

    def _augmentations(self, train: bool) -> A.Compose:
        # Base augmentations (augmentations that are applied to all splits of the data)
        augs = self._base_augmentations
        # Add training augmentations on top of base augmentations
        if train and self.data_aug:
            augs.extend(self.train_transforms)
        # ToTensorV2 should always be the final op in the albumenations pipeline
        augs.append(ToTensorV2())
        return A.Compose(augs)

    @implements(LightningDataModule)
    def setup(self, stage: str | None = None) -> None:
        dataset, base_dir = em.celeba(
            download_dir=self.data_dir,
            label=self.y_label,
            sens_attr=self.s_label,
            download=False,
            check_integrity=True,
        )

        train_transform = self._augmentations(train=True)
        test_transform = self._augmentations(train=False)

        assert dataset is not None
        all_data = TiWrapper(
            emvi.TorchImageDataset(
                data=dataset.load(), root=base_dir, transform=None, target_transform=None
            )
        )

        num_train_val, _ = self._get_splits(int(len(all_data)), self.test_split)
        num_train, num_val = self._get_splits(num_train_val, self.val_split)

        g_cpu = torch.Generator()
        g_cpu = g_cpu.manual_seed(self.seed)
        train_data, val_data, test_data = random_split(
            all_data,
            lengths=(
                num_train,
                num_val,
                len(all_data) - num_train - num_val,
            ),
            generator=g_cpu,
        )
        self._train_data = AlbumentationsDataset(dataset=train_data, transform=train_transform)
        self._val_data = AlbumentationsDataset(dataset=val_data, transform=test_transform)
        self._test_data = AlbumentationsDataset(dataset=test_data, transform=test_transform)
