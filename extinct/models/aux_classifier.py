from typing import List, Tuple

from kit import implements
import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim

from extinct.datamodules.structures import DataBatch


class AuxClassifier(pl.LightningModule):
    def __init__(
        self,
        enc: nn.Module,
        classifier: nn.Module,
        lr: float,
        weight_decay: float,
        lr_gamma: float,
    ):
        super().__init__()
        self.enc = enc
        self.enc.eval()
        self.clf = classifier
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.lr_gamma = lr_gamma
        self._clf_loss = nn.BCEWithLogitsLoss(reduction="mean")

    def training_step(self, batch: DataBatch, batch_idx: int) -> Tensor:
        logits = self(batch.x)
        return self._clf_loss(logits, batch.y)

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            z = self.enc(x)
        return self.clf(z)

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler.ExponentialLR]]:
        opt = optim.AdamW(
            self.clf.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        sched = optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=self.lr_gamma)
        return [opt], [sched]
