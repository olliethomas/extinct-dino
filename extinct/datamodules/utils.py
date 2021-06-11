from __future__ import annotations
from typing import TYPE_CHECKING, Union, cast

import ethicml.vision as emvi
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Subset

if TYPE_CHECKING:
    from extinct.datamodules import TiWrapper

    _Dataset = Union[emvi.TorchImageDataset, TiWrapper]
    ExtractableDataset = Union[ConcatDataset[_Dataset], _Dataset]

__all__ = ["extract_labels_from_dataset"]


def extract_labels_from_dataset(dataset: ExtractableDataset) -> tuple[Tensor, Tensor]:
    def _extract(dataset: _Dataset) -> tuple[Tensor, Tensor]:
        if isinstance(dataset, Subset):
            _s = cast(Tensor, dataset.dataset.s[dataset.indices])  # type: ignore
            _y = cast(Tensor, dataset.dataset.y[dataset.indices])  # type: ignore
        else:
            _s = dataset.s
            _y = dataset.y
        return _s, _y

    if isinstance(dataset, ConcatDataset):
        s_all_ls, y_all_ls = [], []
        for _dataset in dataset.datasets:
            s, y = _extract(_dataset)  # type: ignore
            s_all_ls.append(s)
            y_all_ls.append(y)
        s_all = torch.cat(s_all_ls, dim=0)
        y_all = torch.cat(y_all_ls, dim=0)
    else:
        s_all, y_all = _extract(dataset)  # type: ignore
    return s_all, y_all
