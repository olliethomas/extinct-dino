import pytorch_lightning as pl
from fair_bolts.datasets.ethicml_datasets import DataBatch
from kit import implements
from torch import Tensor, optim


class DinoModel(pl.LightningModule):
    def __init__(self, lr: float, weight_decay: float):
        super().__init__()
        self.learning_rate = lr
        self.weight_decay = weight_decay

    def training_step(self, batch: DataBatch, batch_idx: int) -> Tensor:
        return self(batch.x)

    def forward(self, x: Tensor) -> Tensor:
        return x

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> optim.Optimizer:
        return optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
