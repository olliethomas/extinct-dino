from __future__ import annotations
from typing import Any

from bolts.models import ErmBaseline
from torch import nn

from extinct.components.models.predefined import Mp64x64Net

__all__ = ["CelebaErm"]


class CelebaErm(ErmBaseline):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        lr_gamma: float,
        batch_norm: bool = True,
    ) -> None:
        enc = Mp64x64Net(batch_norm=batch_norm, in_chans=3, target_dim=10)
        clf = nn.Linear(10, 2)
        super().__init__(
            enc=enc,
            clf=clf,
            lr=lr,
            weight_decay=weight_decay,
            lr_gamma=lr_gamma,
        )

    def build(self, *args: Any, **kwargs: Any) -> None:
        """Not needed."""
