from typing import Dict

from kit import implements
import pytorch_lightning as pl
from torch import Tensor, nn, optim
import torch.nn.functional as F
import torchmetrics

from extinct.datamodules.structures import DataBatch
from extinct.models.predefined import Mp64x64Net


class KCBaseline(pl.LightningModule):
    def __init__(self, lr: float, weight_decay: float, batch_norm: bool):
        super().__init__()
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.net = Mp64x64Net(batch_norm=batch_norm, in_chans=3, target_dim=1)

        self.test_acc = torchmetrics.Accuracy()

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

    @implements(pl.LightningModule)
    def test_step(self, batch: DataBatch, batch_idx: int) -> Dict[str, Tensor]:
        out = self(batch.x)
        loss = F.binary_cross_entropy_with_logits(out, batch.y.float())
        acc = self.test_acc(out.sigmoid(), batch.y)
        self.log_dict(
            {
                f"test/loss": loss.item(),
                f"test/acc": acc,
            }
        )

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
