from __future__ import annotations
from typing import Callable, cast

from fair_bolts.datasets.ethicml_datasets import DataBatch
from kit import implements
import pytorch_lightning as pl
from torch import Tensor, nn, optim

from . import vit
from .head import DINOHead
from .loss import DINOLoss
from .utils import MultiCropWrapper, cosine_scheduler, get_params_groups

__all__ = ["DinoModel"]


class DINO(pl.LightningModule):
    _loss_fn: DINOLoss
    student: MultiCropWrapper
    teacher: MultiCropWrapper

    def __init__(
        self,
        lr: float = 5.0e-4,
        warmup_iters: int = 10,
        weight_decay: float = 4.0e-2,
        min_lr: float = 1.0e-6,
        weight_decay_end: float = 0.4,
        freeze_last_layer: int = 1,
        arch: vit.VitArch = vit.VitArch.small,
        patch_size: int = 16,
        out_dim: int = 65_536,
        norm_last_layer: bool = True,
        use_bn_in_head: bool = False,
        momentum_teacher: float = 0.996,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_iters: int = 30,
    ) -> None:
        super().__init__()
        self.learning_rate = lr
        self.warmup_iters = warmup_iters
        self.weight_decay = weight_decay
        self.weight_decay_end = weight_decay_end
        self.min_lr = min_lr
        self.freeze_last_layer = freeze_last_layer
        self.patch_size = patch_size
        self.out_dim = out_dim
        self.norm_last_layer = norm_last_layer
        self.use_bn_in_head = use_bn_in_head
        self.momentum_teacher = momentum_teacher
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp_iters = warmup_teacher_temp_iters

        self._arch_fn = cast(
            Callable[[int], vit.VisionTransformer], getattr(vit, f"vit_{arch.value}")
        )

    def _build(self, train_iters: int, batch_size_per_gpu: int, num_gpus: int) -> None:
        for net in ("student", "teacher"):
            backbone = self._arch_fn(self.patch_size)
            embed_dim = backbone.embed_dim
            norm_last_layer = (net == "teacher") or self.norm_last_layer
            head = DINOHead(
                embed_dim,
                self.out_dim,
                use_bn=self.use_bn_in_head,
                norm_last_layer=norm_last_layer,
            )
            setattr(self, net, MultiCropWrapper(backbone=backbone, head=head))

        self._loss_fn = DINOLoss(
            out_dim=self.out_dim,
            warmup_teacher_temp=self.teacher_temp,
            teacher_temp=self.teacher_temp,
            warmup_teacher_temp_iters=self.warmup_teacher_temp_iters,
            total_iters=train_iters,
        )

        self.lr_schedule = cosine_scheduler(
            base_value=self.lr * (batch_size_per_gpu * num_gpus) / 256.0,  # linear scaling rule
            final_value=self.min_lr,
            total_iters=train_iters,
            warmup_iters=self.warmup_iters,
        )
        self.wd_schedule = cosine_scheduler(
            base_value=self.weight_decay,
            final_value=self.weight_decay_end,
            total_iters=train_iters,
        )
        self.momentum_schedule = cosine_scheduler(
            base_value=self.momentum_teacher,
            final_value=1,
            total_iters=train_iters,
        )

    @implements(pl.LightningModule)
    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamW(
            get_params_groups(self.student), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    @torch.no_grad()
    def _update_momentum_teacher(self, train_itr: int) -> None:
        """
        Momentum update of the teacher network
        """
        em = self.momentum_schedule[train_itr]  # momentum parameter
        for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)

    def _cancel_gradients_last_layer(self, epoch: int) -> None:
        if epoch >= self.freeze_last_layer:
            return
        for n, p in self.student.named_parameters():
            if "last_layer" in n:
                p.grad = None

    @implements(pl.LightningModule)
    def training_step(self, batch: DataBatch, batch_idx: int) -> Tensor:
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            param_group["lr"] = self.lr_schedule[batch_idx]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.wd_schedule[batch_idx]

        teacher_output = self.teacher(
            batch.x[:2]
        )  # only the 2 global views pass through the teacher
        student_output = self.student(batch.x)
        return self._loss_fn(student_output, teacher_output, batch_idx)

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: optim.Optimizer,
        optimizer_idx: int,
        optimizer_closure: Callable | None,
        on_tpu: bool,
        using_native_amp: bool,
        using_lbfgs: bool,
    ) -> None:
        # Keep the output layer fixed until the epoch-threshold has been reached
        # Typicacally doing so during the first epoch helps training.
        self._cancel_gradients_last_layer(epoch=epoch)
        # Update the student's parameters using the DINO loss
        super().optimizer_step(
            epoch=epoch,
            batch_idx=batch_idx,
            optimizer=optimizer,
            optimizer_idx=optimizer_idx,
            optimizer_closure=optimizer_closure,
            on_tpu=on_tpu,
            using_native_amp=using_native_amp,
            using_lbfgs=using_lbfgs,
        )
        # Update the teacher network via EMA of the student's weights
        self._update_momentum_teacher(train_itr=batch_idx)

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.student.backbone(x)
