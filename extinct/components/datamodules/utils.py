from __future__ import annotations
from functools import lru_cache
from typing import Union, cast

import ethicml.vision as emvi
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Subset

from .structures import AlbumentationsDataset, TiWrapper

__all__ = ["extract_labels_from_dataset"]

_Dataset = Union[emvi.TorchImageDataset, TiWrapper]
ExtractableDataset = Union[ConcatDataset[_Dataset], _Dataset, AlbumentationsDataset]


@lru_cache(typed=True)
def extract_labels_from_dataset(dataset: ExtractableDataset) -> tuple[Tensor, Tensor]:
    def _extract(dataset: _Dataset) -> tuple[Tensor, Tensor]:
        if isinstance(dataset, Subset):
            _s = cast(Tensor, dataset.dataset.s[dataset.indices])  # type: ignore
            _y = cast(Tensor, dataset.dataset.y[dataset.indices])  # type: ignore
        else:
            _s = dataset.s
            _y = dataset.y
        return _s, _y

    try:
        if isinstance(dataset, AlbumentationsDataset):
            dataset = dataset.dataset  # type: ignore
        if isinstance(dataset, (ConcatDataset)):
            s_all_ls, y_all_ls = [], []
            for _dataset in dataset.datasets:
                s, y = _extract(_dataset)  # type: ignore
                s_all_ls.append(s)
                y_all_ls.append(y)
            s_all = torch.cat(s_all_ls, dim=0)
            y_all = torch.cat(y_all_ls, dim=0)
        else:
            s_all, y_all = _extract(dataset)  # type: ignore
    except AttributeError:
        # Resort to the Brute-force approach of iterating over the dataset
        s_all_ls, y_all_ls = [], []
        for batch in dataset:
            s_all_ls.append(batch[1])
            y_all_ls.append(batch[2])
        s_all = torch.cat(s_all_ls, dim=0)
        y_all = torch.cat(y_all_ls, dim=0)
    return s_all, y_all
