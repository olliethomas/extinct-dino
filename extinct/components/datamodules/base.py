from __future__ import annotations
from abc import abstractmethod
from enum import Enum, auto
import logging
import os
from typing import Optional, Sequence

import albumentations as A
from albumentations.pytorch import ToTensorV2
from kit import gcopy, implements
from kit.torch import InfSequentialBatchSampler, StratifiedSampler
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    Sampler,
    SequentialSampler,
)

from extinct.components.datamodules.utils import extract_labels_from_dataset

from .dino import DINOAugmentation

__all__ = [
    "BaseDataModule",
    "VisionDataModule",
    "TrainAugMode",
]


LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


class BaseDataModule(pl.LightningDataModule):
    """Base DataModule of both Tabular and Vision DataModules."""

    _train_data: Dataset
    _val_data: Dataset
    _test_data: Dataset

    def __init__(
        self,
        data_dir: str | None,
        y_dim: int,
        s_dim: int,
        batch_size: int,
        val_pcnt: float,
        test_pcnt: float,
        num_workers: int,
        seed: int,
        persist_workers: bool,
        pin_memory: bool,
        stratified_sampling: bool = False,
        sample_with_replacement: bool = True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_pcnt = val_pcnt
        self.test_pcnt = test_pcnt
        self.seed = seed
        self.persist_workers = persist_workers
        self.pin_memory = pin_memory

        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.y_dim = y_dim
        self.s_dim = s_dim
        self.seed = seed

        self.stratified_sampling = stratified_sampling
        self.sample_with_replacement = sample_with_replacement

    def make_dataloader(
        self,
        ds: Dataset,
        batch_sampler: Sampler[Sequence[int]] | None = None,
    ) -> DataLoader:
        """Make DataLoader."""
        return DataLoader(
            ds,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.persist_workers,
            batch_sampler=batch_sampler,
        )

    def _get_train_data(self, eval_: bool) -> Dataset:
        return self._train_data

    @implements(LightningDataModule)
    def train_dataloader(
        self, batch_size: int | None = None, shuffle: bool = True, eval_: bool = False
    ) -> DataLoader:
        train_data = self._get_train_data(eval_=eval_)
        batch_size = self.batch_size if batch_size is None else batch_size

        if eval_:
            batch_sampler = BatchSampler(
                sampler=SequentialSampler(data_source=train_data),
                batch_size=batch_size,
                drop_last=False,
            )
        elif self.stratified_sampling:
            s_all, y_all = extract_labels_from_dataset(self._train_data)
            group_ids = (y_all * len(s_all.unique()) + s_all).squeeze()
            num_samples_per_group = batch_size // (num_groups := len(group_ids.unique()))
            if self.batch_size % num_groups:
                LOGGER.info(
                    'For stratified sampling, the batch size must be a multiple of the number of groups.Since the batch size is not integer divisible by the number of groups ({num_groups}),the batch size is being reduced to {num_samples_per_group * num_groups}.'
                )

            batch_sampler = StratifiedSampler(
                group_ids.squeeze().tolist(),
                num_samples_per_group=num_samples_per_group,
                replacement=self.sample_with_replacement,
                base_sampler="sequential",
                shuffle=shuffle,
            )
        else:
            batch_sampler = InfSequentialBatchSampler(
                data_source=self._train_data, batch_size=batch_size, shuffle=shuffle  # type: ignore
            )
        return self.make_dataloader(train_data, batch_sampler=batch_sampler)

    @implements(pl.LightningDataModule)
    def val_dataloader(self) -> DataLoader:
        return self.make_dataloader(self._val_data)

    @implements(pl.LightningDataModule)
    def test_dataloader(self) -> DataLoader:
        return self.make_dataloader(self._test_data)


class TrainAugMode(Enum):
    none = auto()
    basic = auto()
    dino = auto()


class VisionDataModule(BaseDataModule):
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
        pin_memory: bool = True,
        aug_mode: TrainAugMode = TrainAugMode.none,
        # Sampling settings
        stratified_sampling: bool = False,
        sample_with_replacement: bool = True,
        # DINO parameters
        global_crops_scale: tuple[float, float] = (0.4, 1.0),
        local_crops_scale: tuple[float, float] = (0.05, 0.4),
        local_crops_number: int = 8,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            val_pcnt=val_split,
            test_pcnt=test_split,
            y_dim=y_dim,
            s_dim=s_dim,
            seed=seed,
            persist_workers=persist_workers,
            pin_memory=pin_memory,
            stratified_sampling=stratified_sampling,
            sample_with_replacement=sample_with_replacement,
        )
        self.aug_mode = aug_mode
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number

        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.y_dim = y_dim
        self.s_dim = s_dim
        self.seed = seed

    @property
    @abstractmethod
    def _base_augmentations(self) -> A.Compose:
        ...

    @property
    @abstractmethod
    def _train_augmentations(self) -> A.Compose:
        ...

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
            # Normalization is handled within DinoAugmentation since it needs to be applied
            # to each crop separately
            augs.append(self._normalization)
        return A.Compose(augs)

    def _get_train_data(self, eval_: bool) -> Dataset:
        train_data = self._train_data
        if self.aug_mode is TrainAugMode.dino:
            dino_eval_transforms = A.Compose(
                [
                    A.RandomResizedCrop(height=224, width=224),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
            train_data = gcopy(train_data, transform=dino_eval_transforms)
        return train_data
