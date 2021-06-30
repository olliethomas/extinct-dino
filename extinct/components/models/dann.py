from bolts.models import Dann as _Dann
import pytorch_lightning as pl

from extinct.components.models.predefined import EmbeddingClf, Encoder

__all__ = ["Dann"]


class Dann(_Dann):
    def __init__(self, lr: float, weight_decay: float, grl_lambda: float = 1.0) -> None:
        enc = Encoder(
            input_shape=(3, 64, 64), initial_hidden_channels=64, levels=3, encoding_dim=128
        )
        adv = EmbeddingClf(encoding_dim=128, out_dim=2)
        clf = EmbeddingClf(encoding_dim=128, out_dim=2)
        super().__init__(
            enc=enc, adv=adv, clf=clf, lr=lr, weight_decay=weight_decay, grl_lambda=grl_lambda
        )

    def build(self, trainer: pl.Trainer, datamodule: pl.LightningDataModule):
        """Function not needed, but currently expected in main.py."""
