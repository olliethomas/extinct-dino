"""CelebA DataModule."""
from __future__ import annotations
from typing import Any, ClassVar, Optional, Tuple
import warnings

import albumentations as A
import ethicml as em
import ethicml.vision as emvi
from kit import implements
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data.dataset import random_split

from extinct.components.datamodules.base import TrainAugMode, VisionDataModule
from extinct.components.datamodules.structures import AlbumentationsDataset, TiWrapper

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
        aug_mode: TrainAugMode = TrainAugMode.none,
        local_crops_number: int = -1,
        global_crops_scale: Tuple[float, float] = (0.4, 1.0),
        local_crops_scale: Tuple[float, float] = (0.05, 0.4),
    ):

        if aug_mode is TrainAugMode.none:
            if local_crops_number > 0:
                warnings.warn(
                    f"Local Crops set to {local_crops_number}, but Augmentation mode "
                    f"is inactive. These values will be ignored."
                )
            if global_crops_scale is not None:
                warnings.warn(
                    f"Global Crops Scale set to {global_crops_scale}, but Augmentation mode "
                    f"is inactive. These values will be ignored."
                )
            if local_crops_scale is not None:
                warnings.warn(
                    f"Global Crops Scale set to {global_crops_scale}, but Augmentation mode "
                    f"is inactive. These values will be ignored."
                )

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
            aug_mode=aug_mode,
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

    @property  # type: ignore[misc]
    @implements(VisionDataModule)
    def _base_augmentations(self) -> A.Compose:
        return A.Compose(
            [
                A.Resize(self.image_size, self.image_size),
                A.CenterCrop(self.image_size, self.image_size),
            ]
        )

    @property  # type: ignore[misc]
    @implements(VisionDataModule)
    def _train_augmentations(self) -> A.Compose:
        # Train-time data augmentations - should be refined further
        return A.Compose(
            [
                A.RandomResizedCrop(height=self.image_size, width=self.image_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(p=0.5),
                A.GaussianBlur(p=0.8),
                A.ToGray(p=0.2),
            ]
        )

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
