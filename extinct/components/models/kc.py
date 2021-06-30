from kit import implements
from torch import Tensor

from extinct.components.datamodules.structures import DataBatch
from extinct.components.models.erm import CelebaErm, ErmBaseline

__all__ = ["CelebaKC"]


class CelebaKC(CelebaErm):
    @implements(ErmBaseline)
    def _get_loss(self, logits: Tensor, batch: DataBatch) -> Tensor:
        return self._loss_fn(input=logits, target=batch.y.float(), instance_weight=batch.iw)
