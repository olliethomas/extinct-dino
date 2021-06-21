from __future__ import annotations
from abc import abstractmethod
from enum import Enum, auto
import logging
from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
from fair_bolts.datamodules.vision_datamodule import VisionBaseDataModule
from kit import implements
from kit.torch import InfSequentialBatchSampler as InfSequentialBatchSampler
from kit.torch import StratifiedSampler
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from extinct.datamodules.utils import extract_labels_from_dataset

from .dino import DINOAugmentation

__all__ = ["VisionDataModule", "TrainAugMode"]


LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class TrainAugMode(Enum):
    none = auto()
    basic = auto()
    dino = auto()


class VisionDataModule(VisionBaseDataModule):
    def __init__(
        self,
        y_dim: int,
        s_dim: int,
        data_dir: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        val_split: float = 0.2,
        test_split: float = 0.2,
        seed: int = 0,
        persist_workers: bool = False,
        stratified_sampling: bool = False,
        sample_with_replacement: bool = True,
        aug_mode: TrainAugMode = TrainAugMode.none,
        # DINO parameters
        global_crops_scale: tuple[float, float] = (0.4, 1.0),
        local_crops_scale: tuple[float, float] = (0.05, 0.4),
        local_crops_number: int = 8,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            y_dim=y_dim,
            s_dim=s_dim,
            seed=seed,
            persist_workers=persist_workers,
        )
        self.stratified_sampling = stratified_sampling
        self.sample_with_replacement = sample_with_replacement
        self.aug_mode = aug_mode
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number

    @property
    @abstractmethod
    def _base_augmentations(self) -> A.Compose:
        return A.Compose([])

    @property
    @abstractmethod
    def _train_augmentations(self) -> A.Compose:
        return A.Compose([])

    @property
    def _normalization(self) -> A.Compose:
        return A.Compose([A.ToFloat(), A.Normalize(), ToTensorV2()])

    def _augmentations(self, train: bool) -> A.Compose:
        # Base augmentations (augmentations that are applied to all splits of the data)
        augs: list[A.ImageOnlyTransform | A.Compose] = [self._base_augmentations]
        # Add training augmentations on top of base augmentations
        if train:
            if self.aug_mode is TrainAugMode.basic:
                augs.append(self._train_augmentations)
            elif self.aug_mode is TrainAugMode.dino:
                augs.append(
                    DINOAugmentation(
                        global_crops_scale=self.global_crops_scale,
                        local_crops_scale=self.local_crops_scale,
                        local_crops_number=self.local_crops_number,
                    )
                )
        if not train or self.aug_mode is not TrainAugMode.dino:
            # Normalization is hadnled within DinoAugmentation since it needs to be applied
            # to each crop separately
            augs.append(self._normalization)
        return A.Compose(augs)

    @implements(LightningDataModule)
    def train_dataloader(self, shuffle: bool = True, eval: bool = False) -> DataLoader:
        if eval:
            batch_sampler = None
        else:
            if self.stratified_sampling:
                s_all, y_all = extract_labels_from_dataset(self._train_data)
                group_ids = (y_all * len(s_all.unique()) + s_all).squeeze()
                num_samples_per_group = self.batch_size // (num_groups := len(group_ids.unique()))
                if self.batch_size % num_groups:
                    LOGGER.info(
                        f"For stratified sampling, the batch size must be a multiple of the number of groups."
                        "Since the batch size is not integer divisible by the number of groups ({num_groups}),"
                        "the batch size is being reduced to {num_samples_per_group * num_groups}."
                    )
                batch_sampler = StratifiedSampler(
                    group_ids.squeeze().tolist(),
                    num_samples_per_group=num_samples_per_group,
                    replacement=self.sample_with_replacement,
                )
            else:
                batch_sampler = InfSequentialBatchSampler(
                    data_source=self._train_data, batch_size=self.batch_size, shuffle=shuffle  # type: ignore
                )
        return DataLoader(
            self._train_data,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.persist_workers,
            batch_sampler=batch_sampler,
        )
