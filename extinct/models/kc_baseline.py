import pytorch_lightning as pl
from kit import implements
from torch import Tensor, nn, optim

import torch.nn.functional as F

from extinct.datamodules.structures import DataBatch
from extinct.models.predefined import Mp64x64Net


class KCBaseline(pl.LightningModule):
    def __init__(self, lr: float, weight_decay: float, batch_norm: bool):
        super().__init__()
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.net = Mp64x64Net(batch_norm=batch_norm)(in_chans=3, target_dim=1)

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> optim.Optimizer:
        return optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    @implements(pl.LightningModule)
    def training_step(self, batch: DataBatch, batch_idx: int) -> Tensor:
        out = self(batch.x)
        return F.binary_cross_entropy_with_logits(
            input=out, target=batch.y.float(), weight=batch.iw
        )

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
