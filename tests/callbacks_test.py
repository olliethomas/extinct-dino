from __future__ import annotations
from typing import Any

from bolts.datamodules.base_datamodule import BaseDataModule
from kit import implements
from kit.pl import IterationBasedProgBar
from kit.torch import InfSequentialBatchSampler
import pytorch_lightning as pl
import torch
from torch import Tensor, optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


class _DummyModel(pl.LightningModule):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.linear_proj = nn.Linear(in_channels, 1)
        self.val_losses: list[Tensor] = []

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> optim.Optimizer:
        return optim.AdamW(self.parameters())

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.linear_proj(x)

    @implements(pl.LightningModule)
    def training_step(self, batch: tuple[Tensor, Tensor], batch_index: int) -> Tensor:
        x, y = batch
        return (self(x) - y).mean()

    @implements(pl.LightningModule)
    def validation_step(self, batch: tuple[Tensor, Tensor], batch_index: int) -> dict[str, Tensor]:
        x, y = batch
        logits = self(x)
        return {"logits": logits, "y": y}

    @implements(pl.LightningModule)
    def validation_epoch_end(self, output_results: list[dict[str, Tensor]]) -> None:
        logits_all = torch.cat([batch["logits"] for batch in output_results])
        y = torch.cat([batch["y"] for batch in output_results])
        epoch_loss = (logits_all - y).mean().item()
        self.log_dict({"val/loss": epoch_loss})
        self.val_losses.append(epoch_loss)


class _DummyDM(BaseDataModule):
    @staticmethod
    def _generate_data(num_samples: int) -> TensorDataset:
        x = torch.randn(num_samples, 1)
        y = x.squeeze() * 0.7 + 0.4
        return TensorDataset(x, y)

    @property  # type: ignore[misc]
    @implements(BaseDataModule)
    def train_data(self) -> Dataset:
        return self._generate_data(25)

    @property  # type: ignore[misc]
    @implements(BaseDataModule)
    def val_data(self) -> Dataset:
        return self._generate_data(10)

    @property  # type: ignore[misc]
    @implements(BaseDataModule)
    def test_data(self) -> Dataset:
        return self._generate_data(10)

    @implements(pl.LightningDataModule)
    def train_dataloader(self) -> DataLoader:
        dl_kwargs: dict[str, Any] = {
            "shuffle": False,
            "drop_last": False,
        }
        batch_sampler = InfSequentialBatchSampler(self.train_data, batch_size=self.batch_size)
        dl_kwargs["batch_sampler"] = batch_sampler
        return DataLoader(
            self.train_data,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.persist_workers,
            **dl_kwargs,
        )


def test_progbar() -> None:
    dm = _DummyDM(
        batch_size=8, val_split=1, test_split=1, num_workers=0, seed=1, persist_workers=False
    )
    model = _DummyModel(1)
    max_steps = 9
    bar = IterationBasedProgBar()
    trainer = pl.Trainer(
        max_steps=max_steps,
        num_sanity_val_steps=0,
        callbacks=[bar],
        check_val_every_n_epoch=10000,
        val_check_interval=1,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
