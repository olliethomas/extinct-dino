from __future__ import annotations
from enum import Enum, auto
import itertools
from typing import Any, NamedTuple, Union

import ethicml as em
from kit import implements
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim
import torchmetrics

from extinct.components.callbacks.logging import ImageLogger
from extinct.components.datamodules import VisionDataModule
from extinct.components.datamodules.structures import DataBatch
from extinct.components.models import ModelBase
from extinct.components.models.predefined import Decoder, EmbeddingClf, Encoder

__all__ = ["LaftrBaseline"]


class ModelOut(NamedTuple):
    y: Tensor
    z: Tensor
    s: Tensor
    x: Tensor


class DecoderOutAct(Enum):
    none = auto()
    tanh = auto()


class FairnessType(Enum):
    DP = auto()
    EO = auto()
    EqOp = auto()


class LaftrBaseline(ModelBase):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        lr_gamma: float,
        disc_steps: int,
        fairness: FairnessType,
        recon_weight: float,
        clf_weight: float,
        adv_weight: float,
        encoding_dim: int = 128,
        init_hidden_channels: int = 64,
        levels: int = 3,
        image_logging_freq: Union[int, float] = 0.1,
        decoder_out_act: DecoderOutAct = DecoderOutAct.none,
    ) -> None:
        super().__init__()

        self._clf_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self._recon_loss = nn.L1Loss(reduction="mean")
        self._adv_clf_loss = nn.L1Loss(reduction="mean")
        self.encoding_dim = encoding_dim
        self.decoder_out_act = decoder_out_act
        self.init_hidden_channels = init_hidden_channels
        self.levels = levels

        self.disc_steps = disc_steps
        self.fairness = fairness
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.clf_weight = clf_weight
        self.adv_weight = adv_weight
        self.recon_weight = recon_weight

        self.test_acc = torchmetrics.Accuracy()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        if isinstance(image_logging_freq, float) and not (0 <= image_logging_freq <= 1):
            raise ValueError(
                "If image_logging_freq is of type 'float' its value must be between 0 and 1,"
                " corresponding to the fraction of the total number of steps after which to log."
            )
        self.image_logging_freq = image_logging_freq

    def build(self, datamodule: VisionDataModule, trainer: pl.Trainer) -> None:
        input_shape = datamodule.input_size
        self.enc = Encoder(
            input_shape=input_shape,
            initial_hidden_channels=self.init_hidden_channels,
            levels=self.levels,
            encoding_dim=self.encoding_dim,
        )
        decoder_out_act_fn = (
            nn.Identity() if self.decoder_out_act is DecoderOutAct.none else nn.Tanh()
        )
        self.dec = Decoder(
            input_size=input_shape,
            initial_hidden_channels=self.init_hidden_channels,
            levels=self.levels,
            encoding_dim=self.encoding_dim + 1,
            decoder_out_act=decoder_out_act_fn,
        )
        self.adv = EmbeddingClf(encoding_dim=self.encoding_dim)
        self.clf = EmbeddingClf(encoding_dim=self.encoding_dim)

        self.laftr_params = itertools.chain(
            [*self.enc.parameters(), *self.dec.parameters(), *self.clf.parameters()]
        )
        self.adv_params = self.adv.parameters()
        image_logging_freq = (
            self.image_logging_freq
            if isinstance(self.image_logging_freq, int)
            else self.image_logging_freq * trainer.max_steps
        )
        image_logger = ImageLogger(
            logging_freq=image_logging_freq, norm_values=datamodule.norm_values
        )
        trainer.callbacks.append(image_logger)

    def _adv_loss(self, s_pred: Tensor, batch: DataBatch) -> Tensor:
        # For Demographic Parity, for EqOpp is a different loss term.
        if self.fairness is FairnessType.DP:
            s0 = self._adv_clf_loss(s_pred[batch.s == 0], batch.s[batch.s == 0])
            s1 = self._adv_clf_loss(s_pred[batch.s == 1], batch.s[batch.s == 1])
            loss = (s0 + s1) / 2
        elif self.fairness is FairnessType.EO:
            loss = torch.tensor(0.0).to(self.device)
            for s, y in itertools.product([0, 1], repeat=2):
                if len(batch.s[(batch.s == s) & (batch.y == y)]) > 0:
                    loss += self._adv_clf_loss(
                        s_pred[(batch.s == s) & (batch.y == y)],
                        batch.s[(batch.s == s) & (batch.y == y)],
                    )
            loss = 2 - loss
        elif self.fairness is FairnessType.EqOp:
            # TODO: How to best handle this if no +ve samples in the batch?
            loss = torch.tensor(0.0).to(self.device)
            for s in (0, 1):
                if len(batch.s[(batch.s == s) & (batch.y == 1)]) > 0:
                    loss += self._adv_clf_loss(
                        s_pred[(batch.s == s) & (batch.y == 1)],
                        batch.s[(batch.s == s) & (batch.y == 1)],
                    )
            loss = 2 - loss
        else:
            raise RuntimeError("Only DP and EO fairness accepted.")
        return self.adv_weight * loss

    def _inference_step(self, batch: DataBatch, stage: str) -> dict[str, Tensor]:
        model_out = self(batch.x, batch.s)
        laftr_loss = self._laftr_loss(model_out.y, model_out.x, batch)
        adv_loss = self._adv_loss(model_out.s, batch)
        tm_acc = self.val_acc if stage == "val" else self.train_acc
        acc = tm_acc(model_out.y >= 0, batch.y.long())
        self.log_dict(
            {
                f"{stage}/loss": (laftr_loss + adv_loss).item(),
                f"{stage}/model_loss": laftr_loss.item(),
                f"{stage}/adv_loss": adv_loss.item(),
                f"{stage}/{self.target}_acc": acc,
            }
        )
        return {"y": batch.y, "s": batch.s, "preds": model_out.y.sigmoid().round().squeeze(-1)}

    def _inference_epoch_end(self, output_results: list[dict[str, Tensor]], stage: str) -> None:
        all_y = torch.cat([_r["y"] for _r in output_results], 0)
        all_s = torch.cat([_r["s"] for _r in output_results], 0)
        all_preds = torch.cat([_r["preds"] for _r in output_results], 0)

        dt = em.DataTuple(
            x=pd.DataFrame(
                torch.rand_like(all_s, dtype=torch.float32).detach().cpu().numpy(), columns=["x0"]
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

    def _laftr_loss(self, y_pred: Tensor, recon: Tensor, batch: DataBatch) -> Tensor:
        clf_loss = self._clf_loss(y_pred, batch.y)
        recon_loss = self._recon_loss(recon, batch.x)
        return self.clf_weight * clf_loss + self.recon_weight * recon_loss

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[optim.lr_scheduler.ExponentialLR]]:
        opt_laftr = optim.AdamW(self.laftr_params, lr=self.lr, weight_decay=self.weight_decay)
        opt_adv = optim.AdamW(self.adv_params, lr=self.lr, weight_decay=self.weight_decay)

        sched_laftr = optim.lr_scheduler.ExponentialLR(optimizer=opt_laftr, gamma=self.lr_gamma)
        sched_adv = optim.lr_scheduler.ExponentialLR(optimizer=opt_adv, gamma=self.lr_gamma)

        return [opt_laftr, opt_adv], [sched_laftr, sched_adv]

    @implements(pl.LightningModule)
    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_idx: int,
        optimizer_closure: Any,
        on_tpu: bool,
        using_native_amp: bool,
        using_lbfgs: bool,
    ) -> None:
        # update main model every N steps
        if optimizer_idx == 0 and (batch_idx + 1) % self.disc_steps == 0:
            optimizer.step(closure=optimizer_closure)
        if optimizer_idx == 1:  # update discriminator opt every step
            optimizer.step(closure=optimizer_closure)

    @implements(pl.LightningModule)
    def test_step(self, batch: DataBatch, batch_idx: int) -> dict[str, Tensor]:
        return self._inference_step(batch=batch, stage="test")

    @implements(pl.LightningModule)
    def test_epoch_end(self, output_results: list[dict[str, Tensor]]) -> None:
        self._inference_epoch_end(output_results=output_results, stage="test")

    @implements(pl.LightningModule)
    def training_step(self, batch: DataBatch, batch_idx: int, optimizer_idx: int) -> Tensor:
        if optimizer_idx == 0:
            # Main model update
            self.set_requires_grad(self.adv, requires_grad=False)
            model_out = self(batch.x, batch.s)
            laftr_loss = self._laftr_loss(model_out.y, model_out.x, batch)
            adv_loss = self._adv_loss(model_out.s, batch)
            acc = self.train_acc(model_out.y >= 0, batch.y.long())
            self.log_dict(
                {
                    f"train/loss": (laftr_loss + adv_loss).item(),
                    f"train/model_loss": laftr_loss.item(),
                    f"train/acc": acc,
                }
            )
            return laftr_loss + adv_loss
        elif optimizer_idx == 1:
            # Adversarial update
            self.set_requires_grad([self.enc, self.dec, self.clf], requires_grad=False)
            self.set_requires_grad(self.adv, requires_grad=True)
            model_out = self(batch.x, batch.s)
            adv_loss = self._adv_loss(model_out.s, batch)
            laftr_loss = self._laftr_loss(model_out.y, model_out.x, batch)
            acc = self.train_acc(model_out.y >= 0, batch.y.long())
            self.log_dict(
                {
                    f"train/loss": (laftr_loss + adv_loss).item(),
                    f"train/adv_loss": adv_loss.item(),
                    f"train/acc": acc,
                }
            )
            return -(laftr_loss + adv_loss)
        else:
            raise RuntimeError("There should only be 2 optimizers, but 3rd received.")

    @implements(pl.LightningModule)
    def validation_step(self, batch: DataBatch, batch_idx: int) -> dict[str, Tensor]:
        return self._inference_step(batch=batch, stage="val")

    @implements(pl.LightningModule)
    def validation_epoch_end(self, output_results: list[dict[str, Tensor]]) -> None:
        self._inference_epoch_end(output_results=output_results, stage="val")

    @implements(nn.Module)
    def forward(self, x: Tensor, s: Tensor) -> ModelOut:
        embedding = self.enc(x)
        y_pred = self.clf(embedding)
        s_pred = self.adv(embedding)
        recon = self.dec(embedding, s)
        return ModelOut(y=y_pred, z=embedding, x=recon, s=s_pred)

    def reconstruct(self, x: Tensor) -> Tensor:
        return self.dec(self.enc(x))

    @staticmethod
    def set_requires_grad(nets: nn.Module | list[nn.Module], requires_grad: bool) -> None:
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad
