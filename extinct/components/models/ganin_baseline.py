from bolts.models import Dann

from extinct.components.models.predefined import EmbeddingClf, Encoder

__all__ = ["Ganin"]


class Ganin(Dann):
    def __init__(self, lr: float, weight_decay: float, grl_lambda: float = 1.0):
        enc = Encoder(
            input_shape=(3, 64, 64), initial_hidden_channels=64, levels=3, encoding_dim=128
        )
        adv = EmbeddingClf(encoding_dim=128)
        clf = EmbeddingClf(encoding_dim=128)
        super(Ganin, self).__init__(
            enc=enc, adv=adv, clf=clf, lr=lr, weight_decay=weight_decay, grl_lambda=grl_lambda
        )

    def build(self, *args, **kwargs):
        """Function not needed, but currently expected in main.py."""
