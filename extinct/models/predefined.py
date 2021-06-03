from typing import List

from torch import Tensor, nn


class Mp64x64Net(nn.Module):
    def __init__(self, batch_norm: bool, in_chans: int, target_dim: int):
        super().__init__()
        self.batch_norm = batch_norm
        self.net = self._build(in_chans=in_chans, target_dim=target_dim)

    def _conv_block(
        self, in_chans: int, out_dim: int, kernel_size: int, stride: int, padding: int
    ) -> List[nn.Module]:
        _block: List[nn.Module] = []
        _block += [
            nn.Conv2d(in_chans, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        if self.batch_norm:
            _block += [nn.BatchNorm2d(out_dim)]
        _block += [nn.LeakyReLU()]
        return _block

    def _build(self, in_chans: int, target_dim: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        layers.extend(self._conv_block(in_chans, 64, 5, 1, 0))
        layers += [nn.MaxPool2d(2, 2)]

        layers.extend(self._conv_block(64, 128, 3, 1, 1))
        layers += [nn.MaxPool2d(2, 2)]

        layers.extend(self._conv_block(128, 128, 3, 1, 1))
        layers += [nn.MaxPool2d(2, 2)]

        layers.extend(self._conv_block(128, 256, 3, 1, 1))
        layers += [nn.MaxPool2d(2, 2)]

        layers.extend(self._conv_block(256, 512, 3, 1, 1))
        layers += [nn.MaxPool2d(2, 2)]

        layers += [nn.Flatten()]
        layers += [nn.Linear(512, target_dim)]

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
