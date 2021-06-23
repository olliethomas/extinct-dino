from __future__ import annotations

from kit import implements
import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim

from extinct.components.models import ErmBaseline

__all__ = ["FineTuner"]


class FineTuner(ErmBaseline):
    def reset_parameters(self) -> None:
        self.clf.apply(self._maybe_reset_parameters)

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            z = self.enc(x)
        return self.clf(z)

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[optim.lr_scheduler.ExponentialLR]]:
        opt = optim.AdamW(
            self.clf.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        sched = optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=self.lr_gamma)
        return [opt], [sched]
