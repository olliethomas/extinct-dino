from __future__ import annotations
from typing import Any, Callable, cast

from kit import implements
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from extinct.datamodules import DataBatch, Stage, VisionDataModule
from extinct.models.base import ModelBase
from extinct.utils.callbacks import IterationBasedProgBar

from . import vit
from .eval import KNN, DatasetEncoderRunner, DINOLinearClassifier, EvalMethod
from .head import DINOHead, MultiCropWrapper
from .loss import DINOLoss
from .utils import cosine_scheduler, get_params_groups

__all__ = ["DINO"]


class DINO(ModelBase):
    _loss_fn: DINOLoss
    student: MultiCropWrapper
    teacher: MultiCropWrapper
    eval_trainer: pl.Trainer
    lr_schedule: np.ndarray
    wd_schedule: np.ndarray
    momentum_schedule: np.ndarray
    datamodule: VisionDataModule
    eval_clf: DINOLinearClassifier | KNN

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
        eval_method: EvalMethod = EvalMethod.lin_clf,
        num_eval_blocks: int = 1,
        lr_eval: float = 1.0e-4,
        lin_clf_steps: int = 1000,
    ) -> None:
        """
        Args:
            num_eval_blocks: Concatenate [CLS] tokens for the `n` last blocks.
            We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.
        """
        super().__init__()
        self.learning_rate = lr
        self.num_eval_blocks = num_eval_blocks
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
        self.lr_eval = lr_eval
        self.eval_method = eval_method
        self.lin_clf_steps = lin_clf_steps

        self._arch_fn = cast(
            Callable[[int], vit.VisionTransformer], getattr(vit, f"vit_{arch.name}")
        )

    @property
    def enc(self) -> vit.VisionTransformer:
        # We define an encoder-extracting method for consistency with the other models -
        # fit_and_test, for instance, expects a model to comprise of two parts: enc and clf.
        return self.student.backbone

    def _encode_dataset(self, stage: Stage) -> DataBatch:
        # It's not strictly necessary to disable shuffling but pytorch-lightning complains if its
        # enabled during 'testing'
        dl_kwargs = dict(eval=True) if stage == "train" else {}
        # Sampler needs to be set to None, meaning the default sequential/batch sampler combination
        # is used, so that the full dataset is encoded (with no duplicates)
        dataloader = cast(DataLoader, getattr(self.datamodule, f"{stage}_dataloader")(**dl_kwargs))
        # Encode the dataset
        dataset_encoder = DatasetEncoderRunner(model=self.student.backbone)
        print(f"{list(dataset_encoder.parameters())[0].device=}")
        self.eval_trainer.test(
            dataset_encoder,
            test_dataloaders=dataloader,
            verbose=False,
        )
        # Extract the encodings/associated labels from the dataset encoder
        return dataset_encoder.encoded_dataset

    @implements(ModelBase)
    def build(self, datamodule: VisionDataModule, trainer: pl.Trainer) -> None:
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
            warmup_teacher_temp_iters=min(trainer.max_steps - 1, self.warmup_teacher_temp_iters),  # type: ignore
            total_iters=trainer.max_steps,  # type: ignore
            num_crops=datamodule.local_crops_number + 2,
        )

        self.lr_schedule = cosine_scheduler(
            base_value=self.learning_rate * datamodule.batch_size / 256.0,  # linear scaling rule
            final_value=self.min_lr,
            total_iters=trainer.max_steps,  # type: ignore
            warmup_iters=min(trainer.max_steps - 1, self.warmup_iters),  # type: ignore
        )
        self.wd_schedule = cosine_scheduler(
            base_value=self.weight_decay,
            final_value=self.weight_decay_end,
            total_iters=trainer.max_steps,  # type: ignore
        )
        self.momentum_schedule = cosine_scheduler(
            base_value=self.momentum_teacher,
            final_value=1,
            total_iters=trainer.max_steps,  # type: ignore
        )
        self.eval_trainer = pl.Trainer(
            gpus=trainer.gpus,
            max_steps=self.lin_clf_steps,
            distributed_backend=trainer.distributed_backend,
            callbacks=[IterationBasedProgBar()],
            num_sanity_val_steps=0,
        )
        self.datamodule = datamodule

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

    def _cancel_gradients_last_layer(self, train_itr: int) -> None:
        if train_itr >= self.freeze_last_layer:
            return
        for n, p in self.student.named_parameters():
            if "last_layer" in n:
                p.grad = None

    def _get_loss(self, batch: DataBatch, batch_idx: int) -> Tensor:
        teacher_output = self.teacher(
            batch.x[:2]
        )  # only the 2 global views pass through the teacher
        student_output = self.student(batch.x)
        return self._loss_fn(student_output, teacher_output, batch_idx)

    @implements(pl.LightningModule)
    def training_step(self, batch: DataBatch, batch_idx: int) -> Tensor:
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            param_group["lr"] = self.lr_schedule[batch_idx]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.wd_schedule[batch_idx]

        return self._get_loss(batch=batch, batch_idx=batch_idx)

    @implements(pl.LightningModule)
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
        self._cancel_gradients_last_layer(train_itr=batch_idx)
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

    @implements(pl.LightningModule)
    def on_validation_start(self):
        self._on_inference_start()
        super().on_validation_start()

    @implements(pl.LightningModule)
    def on_test_start(self):
        self._on_inference_start()
        super().on_test_start()

    def _on_inference_start(self) -> None:
        if self.eval_method is EvalMethod.lin_clf:
            self.eval_clf = DINOLinearClassifier(
                enc=self.student.backbone,
                target_dim=self.datamodule.y_dim,
                max_steps=self.lin_clf_steps,
                weight_decay=0,
                lr=self.lr_eval,
            )
            self.eval_clf.target = self.target
            print(f"{list(self.eval_clf.parameters())[0].device=}")
            self.eval_trainer.fit(
                self.eval_clf, train_dataloader=self.datamodule.train_dataloader(eval=True)
            )
        else:
            train_data_encoded = self._encode_dataset(stage="train")
            self.eval_clf = KNN(
                train_features=train_data_encoded.x, train_labels=train_data_encoded.y
            )
        self.eval_clf.target = self.target

    @implements(ModelBase)
    def _inference_step(self, batch: DataBatch, stage: Stage) -> dict[str, Any]:
        self.eval_clf.to(batch.x.device)
        return self.eval_clf._inference_step(batch=batch, stage=stage)

    @implements(ModelBase)
    def _inference_epoch_end(
        self, output_results: list[dict[str, Tensor]], stage: Stage
    ) -> dict[str, Any]:
        return self.eval_clf._inference_epoch_end(output_results=output_results, stage=stage)

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.student(x)
