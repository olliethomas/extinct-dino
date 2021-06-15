from __future__ import annotations
import logging
from typing import Optional
import albumentations as A
from abc import abstractmethod

from fair_bolts.datamodules.vision_datamodule import VisionBaseDataModule
from kit import implements
from kit.torch import InfSequentialBatchSampler as InfSequentialBatchSampler
from kit.torch import StratifiedSampler
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from extinct.datamodules.utils import extract_labels_from_dataset

__all__ = ["VisionDataModule"]


LOGGER = logging.getLogger(__name__.split(".")[-1].upper())


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
        data_aug: bool = False,
    ):
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
        self.data_aug = data_aug

    def _augmentations(self, train: bool) -> A.Compose:
        # Base augmentations (augmentations that are applied to all splits of the data)
        augs = self._base_augmentations
        # Add training augmentations on top of base augmentations
        if train and self.data_aug:
            augs.extend(self.train_transforms)
        # ToTensorV2 should always be the final op in the albumenations pipeline
        augs.append(ToTensorV2(p=1.0))
        return A.Compose(augs)

    @property
    @abstractmethod
    def _base_augmentations(self) -> list[A.BasicTransform]:
        tform_ls = [
            A.ToFloat(max_value=1),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
        return tform_ls

    @property
    @abstractmethod
    def _train_augmentations(self) -> list[A.BasicTransform]:
        return []

    @implements(LightningDataModule)
    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
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
