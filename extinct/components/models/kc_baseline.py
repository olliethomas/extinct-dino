from kit import implements
from torch import Tensor

from extinct.components.datamodules.structures import DataBatch
from extinct.components.models.erm_baseline import CelebaErmBaseline, ErmBaseline

__all__ = ["CelebaKCBaseline"]


class CelebaKCBaseline(CelebaErmBaseline):
    @implements(ErmBaseline)
    def _get_loss(self, logits: Tensor, batch: DataBatch) -> Tensor:
        return self._loss_fn(input=logits, target=batch.y.float(), weight=batch.iw)
