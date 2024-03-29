from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn


class Mp64x64Net(nn.Module):
    def __init__(self, batch_norm: bool, in_chans: int, target_dim: int) -> None:
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
        layers = nn.ModuleList()
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


class View(nn.Module):
    def __init__(self, shape: Tuple[int, ...]) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.view(-1, *self.shape)


def down_conv(
    in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        ),
        nn.GroupNorm(num_groups=1, num_channels=out_channels),
        nn.SiLU(),
    )


def up_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int,
) -> nn.Module:
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        ),
        nn.GroupNorm(num_groups=1, num_channels=out_channels),
        nn.SiLU(),
    )


class Encoder(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        initial_hidden_channels: int,
        levels: int,
        encoding_dim: int,
    ) -> None:
        super().__init__()
        layers = nn.ModuleList()
        c_in, height, width = input_shape
        c_out = initial_hidden_channels

        for level in range(levels):
            if level != 0:
                c_in = c_out
                c_out *= 2
            layers.append(
                nn.Sequential(
                    down_conv(c_in, c_out, kernel_size=3, stride=1, padding=1),
                    down_conv(c_out, c_out, kernel_size=4, stride=2, padding=1),
                )
            )
            height //= 2
            width //= 2

        flattened_size = c_out * height * width
        layers += [nn.Flatten()]
        layers += [nn.Linear(flattened_size, encoding_dim)]

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        initial_hidden_channels: int,
        levels: int,
        encoding_dim: int,
        decoding_dim: int,
        decoder_out_act: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        layers = nn.ModuleList()
        c_in, height, width = input_shape
        c_out = initial_hidden_channels

        for level in range(levels):
            if level != 0:
                c_in = c_out
                c_out *= 2

            layers.append(
                nn.Sequential(
                    # inverted order
                    up_conv(c_out, c_out, kernel_size=4, stride=2, padding=1, output_padding=0),
                    down_conv(c_out, c_in, kernel_size=3, stride=1, padding=1),
                )
            )

            height //= 2
            width //= 2

        flattened_size = c_out * height * width

        layers += [View((c_out, height, width))]
        layers += [nn.Linear(encoding_dim, flattened_size)]
        layers = layers[::-1]
        layers += [nn.Conv2d(input_shape[0], decoding_dim, kernel_size=1, stride=1, padding=0)]

        if decoder_out_act is not None:
            layers += [decoder_out_act]

        self.decoder = nn.Sequential(*layers)

    def forward(self, z: Tensor, s: Tensor) -> Tensor:
        zs = torch.cat([z, s], dim=1)
        return self.decoder(zs)


class EmbeddingClf(nn.Module):
    def __init__(
        self,
        encoding_dim: int,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(encoding_dim, 1))

    def forward(self, z: Tensor) -> Tensor:
        return self.classifier(z)
