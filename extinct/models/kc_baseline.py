from kit import implements
from torch import Tensor

from extinct.datamodules.structures import DataBatch
from extinct.models.erm_baseline import ErmBaseline

__all__ = ["KCBaseline"]


class KCBaseline(ErmBaseline):
    @implements(ErmBaseline)
    def _get_loss(self, logits: Tensor, batch: DataBatch) -> Tensor:
        return self._loss_fn(input=logits, target=batch.y.float(), weight=batch.iw)
