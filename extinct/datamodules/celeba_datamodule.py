"""CelebA DataModule."""
from functools import lru_cache
from typing import Any, Optional

import ethicml as em
from ethicml import implements
import ethicml.vision as emvi
from fair_bolts.datamodules.vision_datamodule import VisionBaseDataModule
import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co, random_split
from torchvision import transforms as TF

from extinct.datamodules.structures import DataBatch


class TiWrapper(Dataset):
    """Wrapper for a Torch Image Datasets."""

    def __init__(self, ti: emvi.TorchImageDataset):
        self.ti = ti
        dt = em.DataTuple(
            x=pd.DataFrame(np.random.randint(0, len(ti.s), size=(len(ti.s), 1)), columns=list('x')),
            s=pd.DataFrame(ti.s.cpu().numpy(), columns=["s"]),
            y=pd.DataFrame(ti.y.cpu().numpy(), columns=["y"]),
        )
        self.iws = em.compute_instance_weights(dt)

    def __getitem__(self, index: int) -> T_co:
        x, s, y = self.ti[index]
        iw = x.new_tensor(self.iws.iloc[index])
        return DataBatch(x=x, s=s.squeeze(), y=y.squeeze(), iw=iw)

    def __len__(self) -> int:
        return len(self.ti)


class CelebaDataModule(VisionBaseDataModule):
    """CelebA Dataset."""

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
        cache_data: bool = False,
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
        )
        self.image_size = image_size
        self.dims = (3, self.image_size, self.image_size)
        self.num_classes = 2
        self.num_sens = 2
        self.y_label = y_label
        self.s_label = s_label
        self.cache_data = cache_data

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

        if self.cache_data:
            all_data.__getitem__ = lru_cache(None)(all_data.__getitem__)

        num_train_val, num_test = self._get_splits(int(len(all_data)), self.test_split)
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
