from __future__ import annotations
from typing import List, Optional, Tuple
from typing_extensions import Literal

import ethicml as em
from kit import implements
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F
import torchmetrics

from extinct.datamodules.structures import DataBatch
from extinct.models.predefined import Mp64x64Net

__all__ = ["ErmBaseline"]


Stage = Literal["train", "val", "test"]


class ErmBaseline(pl.LightningModule):
    def __init__(self, lr: float, weight_decay: float, batch_norm: bool, lr_gamma: float):
        super().__init__()
        self.learning_rate = lr
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.enc = Mp64x64Net(batch_norm=batch_norm, in_chans=3, target_dim=10)
        self.clf = nn.Linear(10, 1)
        self.net = nn.Sequential(self.enc, self.clf)
        self._loss_fn = F.binary_cross_entropy_with_logits

        self.test_acc = torchmetrics.Accuracy()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        self._target: Optional[str] = None

    @property
    def target(self) -> str:
        assert self._target is not None
        return self._target

    @target.setter
    def target(self, target: str) -> None:
        self._target = target

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler.ExponentialLR]]:
        opt = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        sched = optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=self.lr_gamma)
        return [opt], [sched]

    def _get_loss(self, logits: Tensor, batch: DataBatch) -> Tensor:
        return self._loss_fn(input=logits, target=batch.y.float())

    def _inference_step(self, batch: DataBatch, stage: Stage) -> dict[str, Tensor]:
        logits = self(batch.x)
        loss = self._get_loss(logits, batch)
        tm_acc = self.val_acc if stage == "val" else self.test_acc
        acc = tm_acc(logits >= 0, batch.y.long())
        self.log_dict(
            {
                f"{stage}/loss": loss.item(),
                f"{stage}/{self.target}_acc": acc,
            }
        )
        return {"y": batch.y, "s": batch.s, "preds": logits.sigmoid().round().squeeze(-1)}

    def _inference_epoch_end(self, output_results: list[dict[str, Tensor]], stage: Stage) -> None:
        all_y = torch.cat([_r["y"] for _r in output_results], 0)
        all_s = torch.cat([_r["s"] for _r in output_results], 0)
        all_preds = torch.cat([_r["preds"] for _r in output_results], 0)

        dt = em.DataTuple(
            x=pd.DataFrame(
                torch.rand_like(all_s, dtype=float).detach().cpu().numpy(), columns=["x0"]
            ),
            s=pd.DataFrame(all_s.detach().cpu().numpy(), columns=["s"]),
            y=pd.DataFrame(all_y.detach().cpu().numpy(), columns=["y"]),
        )

        results = em.run_metrics(
            predictions=em.Prediction(hard=pd.Series(all_preds.detach().cpu().numpy())),
            actual=dt,
            metrics=[em.Accuracy(), em.RenyiCorrelation(), em.Yanovich()],
            per_sens_metrics=[em.Accuracy(), em.ProbPos(), em.TPR()],
        )

        tm_acc = self.val_acc if stage == "val" else self.test_acc
        acc = tm_acc.compute().item()
        results_dict = {f"{stage}/acc": acc}
        results_dict.update({f"{stage}/{self.target}_{k}": v for k, v in results.items()})

        self.log_dict(results_dict)

    @implements(pl.LightningModule)
    def test_step(self, batch: DataBatch, batch_idx: int) -> dict[str, Tensor]:
        return self._inference_step(batch=batch, stage="test")

    @implements(pl.LightningModule)
    def test_epoch_end(self, output_results: list[dict[str, Tensor]]) -> None:
        self._inference_epoch_end(output_results=output_results, stage="test")

    @implements(pl.LightningModule)
    def training_step(self, batch: DataBatch, batch_idx: int) -> Tensor:
        logits = self(batch.x)
        loss = self._get_loss(logits, batch)
        acc = self.train_acc(logits >= 0, batch.y.long())
        self.log_dict(
            {
                f"train/loss": loss.item(),
                f"train/acc": acc,
            }
        )
        return loss

    @implements(pl.LightningModule)
    def validation_step(self, batch: DataBatch, batch_idx: int) -> dict[str, Tensor]:
        return self._inference_step(batch=batch, stage="val")

    @implements(pl.LightningModule)
    def validation_epoch_end(self, output_results: list[dict[str, Tensor]]) -> None:
        self._inference_epoch_end(output_results=output_results, stage="val")

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
