from __future__ import annotations

from kit import implements
import pytorch_lightning as pl
from torch import Tensor, optim
import torch.nn as nn

from extinct.models.finetuning import FineTuner

from .vit import VisionTransformer

__all__ = ["DINOLinearClassifier"]


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
