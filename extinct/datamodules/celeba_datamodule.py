"""CelebA DataModule."""
from typing import Any, ClassVar, Optional

import ethicml as em
from ethicml import implements
import ethicml.vision as emvi
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data.dataset import random_split
from torchvision import transforms as TF

from extinct.datamodules.base import VisionDataModule
from extinct.datamodules.structures import TiWrapper

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

    @implements(LightningDataModule)
    def setup(self, stage: Optional[str] = None) -> None:
        dataset, base_dir = em.celeba(
            download_dir=self.data_dir,
            label=self.y_label,
            sens_attr=self.s_label,
            download=False,
            check_integrity=True,
        )

        tform_ls = [TF.Resize(self.image_size), TF.CenterCrop(self.image_size)]
        tform_ls.append(TF.ToTensor())
        tform_ls.append(TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = TF.Compose(tform_ls)

        assert dataset is not None
        all_data = TiWrapper(
            emvi.TorchImageDataset(
                data=dataset.load(), root=base_dir, transform=transform, target_transform=None
            )
        )

        num_train_val, _ = self._get_splits(int(len(all_data)), self.test_split)
        num_train, num_val = self._get_splits(num_train_val, self.val_split)

        g_cpu = torch.Generator()
        g_cpu = g_cpu.manual_seed(self.seed)
        self._train_data, self._val_data, self._test_data = random_split(
            all_data,
            lengths=(
                num_train,
                num_val,
                len(all_data) - num_train - num_val,
            ),
            generator=g_cpu,
        )
