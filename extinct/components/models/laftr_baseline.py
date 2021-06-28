from bolts.models.laftr_baseline import Laftr
from torch import  nn

from extinct.components.models.predefined import Decoder, EmbeddingClf, Encoder

__all__ = ["LaftrBaseline"]


class LaftrBaseline(Laftr):
    def __init__(
        self,
        lr: float,
        lr_gamma: float,
        weight_decay: float
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
        adv = EmbeddingClf(encoding_dim=128)
        clf = EmbeddingClf(encoding_dim=128)

        super(LaftrBaseline, self).__init__(enc=enc, dec=dec, adv=adv, clf=clf, lr=lr, lr_gamma=lr_gamma, weight_decay=weight_decay)

