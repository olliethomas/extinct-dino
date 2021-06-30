from enum import Enum, auto
from typing import NamedTuple

from bolts.models import Laftr as _Laftr
import pytorch_lightning as pl
from torch import Tensor, nn

from extinct.components.datamodules import VisionDataModule
from extinct.components.models.predefined import Decoder, EmbeddingClf, Encoder

__all__ = ["Laftr"]


class ModelOut(NamedTuple):
    s: Tensor
    x: Tensor
    y: Tensor
    z: Tensor


class FairnessType(Enum):
    DP = auto()
    EO = auto()
    EqOp = auto()


class Laftr(_Laftr):
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
    ) -> None:
        enc = Encoder(
            input_shape=(3, 64, 64), initial_hidden_channels=64, levels=3, encoding_dim=128
        )
        dec = Decoder(
            input_shape=(3, 64, 64),
            initial_hidden_channels=64,
            levels=3,
            encoding_dim=128 + 1,
            decoding_dim=3,
            decoder_out_act=nn.Tanh(),
        )
        adv = EmbeddingClf(encoding_dim=128, out_dim=1)
        clf = EmbeddingClf(encoding_dim=128, out_dim=2)
        super().__init__(
            enc=enc,
            dec=dec,
            adv=adv,
            clf=clf,
            lr=lr,
            weight_decay=weight_decay,
            lr_gamma=lr_gamma,
            disc_steps=disc_steps,
            fairness=fairness,
            recon_weight=recon_weight,
            clf_weight=clf_weight,
            adv_weight=adv_weight,
        )

    def build(self, datamodule: VisionDataModule, trainer: pl.Trainer) -> None:
        """DM and Trainer not needed."""
