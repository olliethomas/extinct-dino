from __future__ import annotations
from typing import NamedTuple

import ethicml as em
import ethicml.vision as emvi
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

__all__ = ["DataBatch", "TiWrapper"]


class DataBatch(NamedTuple):
    x: Tensor
    s: Tensor
    y: Tensor
    iw: Tensor


class TiWrapper(Dataset):
    """Wrapper for a Torch Image Datasets."""

    def __init__(self, ti: emvi.TorchImageDataset):
        self.ti = ti
        # Pull out the data components for compatibility with the extract_labels function
        self.x = ti.x
        self.s = ti.s
        self.y = ti.y

        dt = em.DataTuple(
            x=pd.DataFrame(np.random.randint(0, len(ti.s), size=(len(ti.s), 1)), columns=list("x")),
            s=pd.DataFrame(ti.s.cpu().numpy(), columns=["s"]),
            y=pd.DataFrame(ti.y.cpu().numpy(), columns=["y"]),
        )
        self.iws = torch.tensor(em.compute_instance_weights(dt)["instance weights"].values)

    def __getitem__(self, index: int) -> DataBatch:
        x, s, y = self.ti[index]
        iw = self.iws[index].clone().detach()
        return DataBatch(x=x, s=s.float(), y=y.float(), iw=iw.unsqueeze(-1))

    def __len__(self) -> int:
        return len(self.ti)
