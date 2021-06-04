from collections import namedtuple
import itertools
from typing import List, Union

from kit import implements
import pytorch_lightning as pl
from torch import Tensor, nn, optim
import torch.nn.functional as F

from extinct.datamodules.structures import DataBatch

__all__ = ["LaftrBaseline"]

from extinct.models.predefined import Decoder, EmbeddingClf, Encoder

ModelOut = namedtuple("ModelOut", ["y", "z", "s", "x"])


class LaftrBaseline(pl.LightningModule):
    def __init__(self, lr: float, weight_decay: float, lr_gamma: float):
        super().__init__()
        self.enc = Encoder(
            input_shape=(3, 64, 64), initial_hidden_channels=64, levels=3, encoding_dim=128
        )
        self.dec = Decoder(
            input_shape=(3, 64, 64),
            initial_hidden_channels=64,
            levels=3,
            encoding_dim=128 + 1,
            decoding_dim=3,
            decoder_out_act=nn.Tanh(),
        )
        self.adv = EmbeddingClf(encoding_dim=128)
        self.clf = EmbeddingClf(encoding_dim=128)

        self.laftr_params = itertools.chain(
            [*self.enc.parameters(), *self.dec.parameters(), *self.clf.parameters()]
        )
        self.adv_params = self.adv.parameters()

        self._clf_loss = F.binary_cross_entropy_with_logits
        self._recon_loss = F.l1_loss
        self._adv_clf_loss = F.l1_loss

        self.lr = lr
        self.weight_decay = weight_decay

    def _laftr_loss(self, y_pred: Tensor, recon: Tensor, batch: DataBatch) -> Tensor:
        clf_loss = self._clf_loss(y_pred, batch.y)
        recon_loss = self._recon_loss(recon, batch.x)
        return clf_loss + recon_loss

    def _adv_loss(self, s_pred: Tensor, batch: DataBatch) -> Tensor:
        # For Demographic Parity, for EqOpp is a different loss term.
        s0 = self._adv_clf_loss(s_pred[batch.s == 0], batch.s[batch.s == 0], reduction="mean")
        s1 = self._adv_clf_loss(s_pred[batch.s == 1], batch.s[batch.s == 1], reduction="mean")
        return (s0 + s1) / 2

    @staticmethod
    def set_requires_grad(nets: Union[nn.Module, List[nn.Module]], requires_grad: bool) -> None:
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    @implements(pl.LightningModule)
    def configure_optimizers(self) -> List[optim.Optimizer]:
        opt_laftr = optim.AdamW(self.laftr_params, lr=self.lr, weight_decay=self.weight_decay)
        opt_adv = optim.AdamW(self.adv_params, lr=self.lr, weight_decay=self.weight_decay)
        return [opt_laftr, opt_adv]

    @implements(pl.LightningModule)
    def training_step(self, batch: DataBatch, batch_idx: int, optimizer_idx: int) -> Tensor:
        if optimizer_idx == 0:
            # Main model update
            self.set_requires_grad(self.adv, requires_grad=False)
            model_out = self(batch.x, batch.s)
            return self._laftr_loss(model_out.y, model_out.x, batch)
        elif optimizer_idx == 1:
            # Adversarial update
            self.set_requires_grad([self.enc, self.dec, self.clf], requires_grad=False)
            self.set_requires_grad(self.adv, requires_grad=True)
            model_out = self(batch.x, batch.s)
            return self._adv_loss(model_out.s, batch)
        else:
            raise RuntimeError("There should only be 2 optimizers, but 3rd received.")

    @implements(nn.Module)
    def forward(self, x: Tensor, s: Tensor) -> ModelOut:
        embedding = self.enc(x)
        y_pred = self.clf(embedding)
        s_pred = self.adv(embedding)
        recon = self.dec(embedding, s)
        return ModelOut(y=y_pred, z=embedding, x=recon, s=s_pred)
