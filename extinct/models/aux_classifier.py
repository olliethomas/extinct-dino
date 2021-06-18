from typing import List, Tuple

from kit import implements
import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim
import torchmetrics

from extinct.models import ErmBaseline


class AuxClassifier(ErmBaseline):
    def __init__(
        self,
        enc: nn.Module,
        classifier: nn.Module,
        lr: float,
        weight_decay: float,
        lr_gamma: float,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, batch_norm=True, lr_gamma=lr_gamma)
        self.enc = enc
        self.enc.eval()
        self.clf = classifier
        self.weights_init(self.clf)
        self._loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

        self.test_acc = torchmetrics.Accuracy()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

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
