from typing import Callable

from torch import Tensor, nn

__all__ = ["MultiCropNet"]

from . import vit
from .head import DINOHead, MultiCropWrapper


class MultiCropNet(nn.Module):
    def __init__(
        self,
        arch_fn: Callable[[int], vit.VisionTransformer],
        patch_size: int,
        norm_last_layer: bool,
        use_bn_in_head: bool,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.backbone = arch_fn(patch_size)
        embed_dim = self.backbone.embed_dim
        self.head = DINOHead(
            embed_dim,
            out_dim,
            use_bn=use_bn_in_head,
            norm_last_layer=norm_last_layer,
        )
        self.net = MultiCropWrapper(backbone=self.backbone, head=self.head)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
