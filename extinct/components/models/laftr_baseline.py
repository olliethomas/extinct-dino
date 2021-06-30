from collections import namedtuple
from enum import Enum

from bolts.models import Laftr
import pytorch_lightning as pl
from torch import nn

from extinct.components.datamodules import VisionDataModule
from extinct.components.models.predefined import Decoder, EmbeddingClf, Encoder

__all__ = ["LaftrBaseline"]


ModelOut = namedtuple("ModelOut", ["y", "z", "s", "x"])
FairnessType = Enum("FairnessType", "DP EO EqOp")


class LaftrBaseline(Laftr):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        lr_gamma: float,
        disc_steps: int,
        fairness: str,
        recon_weight: float,
        clf_weight: float,
        adv_weight: float,
    ) -> None:
        assert fairness in FairnessType._member_names_
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
