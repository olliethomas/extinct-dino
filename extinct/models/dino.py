from fair_bolts.datasets.ethicml_datasets import DataBatch
from kit import implements
import pytorch_lightning as pl
from torch import Tensor, nn, optim

__all__ = ["DinoModel"]


class DinoModel(pl.LightningModule):
    def __init__(self, lr: float, weight_decay: float, lr_gamma: float):
        super().__init__()
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.lr_gamma = lr_gamma

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> optim.Optimizer:
        return optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    @implements(pl.LightningModule)
    def training_step(self, batch: DataBatch, batch_idx: int) -> Tensor:
        return self(batch.x)

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return x
