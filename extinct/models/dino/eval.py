from __future__ import annotations
from enum import Enum, auto
from typing import Any

import ethicml as em
from kit import implements
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor, optim
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from extinct.datamodules import DataBatch, Stage
from extinct.models.finetuning import FineTuner

from .vit import VisionTransformer

__all__ = ["EvalMethod", "DINOLinearClassifier", "DatasetEncoderRunner", "KNN"]


class EvalMethod(Enum):
    knn = auto()
    lin_clf = auto()


class DINOLinearClassifier(FineTuner):
    enc: VisionTransformer

    def __init__(
        self,
        enc: VisionTransformer,
        target_dim: int,
        weight_decay: float,
        lr: float,
        max_steps: int,
        num_eval_blocks: int = 1,
    ) -> None:
        clf = nn.Linear(enc.embed_dim * num_eval_blocks, target_dim)
        super().__init__(enc=enc, clf=clf, lr=lr, weight_decay=weight_decay, lr_gamma=1)
        self.max_steps = max_steps
        self.num_eval_blocks = num_eval_blocks

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[optim.lr_scheduler.CosineAnnealingLR]]:
        opt = optim.SGD(
            self.clf.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        sched = optim.lr_scheduler.CosineAnnealingLR(optimizer=opt, T_max=self.max_steps, eta_min=0)
        return [opt], [sched]

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.clf(self.enc.encode(x, num_eval_blocks=self.num_eval_blocks))


class DatasetEncoderRunner(pl.LightningModule):
    """Wrapper for extractor model."""

    encoded_dataset: DataBatch

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    @implements(pl.LightningModule)
    def test_step(self, batch: DataBatch, batch_idx: int) -> DataBatch:
        return DataBatch(self(batch.x), *batch[1:])

    @implements(pl.LightningModule)
    def test_epoch_end(self, outputs: list[DataBatch]) -> None:
        outputs_t = tuple(zip(*outputs))
        self.encoded_dataset = DataBatch(*(torch.cat(el, dim=0) for el in outputs_t))


class KNN:
    # This implementation is pretty hacky in that it involves duplication of components from
    # ModelBase and ErmBaseline and implements methods defined by the former's interface without inheriting from it
    _target: str | None

    @property
    def target(self) -> str:
        assert self._target is not None
        return self._target

    def __init__(self, train_features: Tensor, train_labels: Tensor, k: int = 5) -> None:
        self.train_features = nn.Parameter(
            F.normalize(train_features.detach(), dim=1, p=2), requires_grad=False
        )
        self.train_labels = nn.Parameter(train_labels.detach(), requires_grad=False)
        self.k = k

        self.test_acc = torchmetrics.Accuracy()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, test_features: Tensor) -> Tensor:
        test_features = F.normalize(test_features, dim=1, p=2)
        similarity = test_features @ self.train_features.t()
        # distances, indices = similarity.topk(k, largest=True, sorted=True)
        return self.train_labels[similarity.argmax()]

    def _inference_step(self, batch: DataBatch, stage: Stage) -> dict[str, Tensor]:
        preds = self.forward(batch.x)
        tm_acc = self.val_acc if stage == "val" else self.test_acc
        tm_acc(preds, batch.y.long())
        return {"y": batch.y, "s": batch.s, "preds": preds}

    def _inference_epoch_end(
        self, output_results: list[dict[str, Tensor]], stage: Stage
    ) -> dict[str, Any]:
        all_y = torch.cat([_r["y"] for _r in output_results], 0)
        all_s = torch.cat([_r["s"] for _r in output_results], 0)
        all_preds = torch.cat([_r["preds"] for _r in output_results], 0)

        dt = em.DataTuple(
            x=pd.DataFrame(
                torch.rand_like(all_s, dtype=torch.float).detach().cpu().numpy(), columns=["x0"]
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
        return results_dict
