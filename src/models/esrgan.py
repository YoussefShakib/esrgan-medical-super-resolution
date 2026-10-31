\
import math
import torch
import torch.nn as nn


class DenseResidualBlock(nn.Module):
    def __init__(self, channels: int = 64, growth_channels: int = 32, residual_scale: float = 0.2) -> None:
        super().__init__()
        self.residual_scale = residual_scale

        self.conv1 = nn.Conv2d(channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, 3, 1, 1)

        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.activation(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.activation(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + x5 * self.residual_scale


class RRDB(nn.Module):
    def __init__(self, channels: int = 64, growth_channels: int = 32, residual_scale: float = 0.2) -> None:
        super().__init__()
        self.block1 = DenseResidualBlock(channels, growth_channels, residual_scale)
        self.block2 = DenseResidualBlock(channels, growth_channels, residual_scale)
        self.block3 = DenseResidualBlock(channels, growth_channels, residual_scale)
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        return x + out * self.residual_scale


class RRDBNet(nn.Module):
    def __init__(
        self,
        scale: int = 4,
        channels: int = 64,
        num_blocks: int = 10,
        growth_channels: int = 32,
    ) -> None:
        super().__init__()
        if scale not in {2, 4, 8}:
            raise ValueError("RRDBNet currently supports scale 2, 4, or 8")

        self.scale = scale
        self.conv_first = nn.Conv2d(3, channels, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(channels, growth_channels) for _ in range(num_blocks)])
        self.conv_body = nn.Conv2d(channels, channels, 3, 1, 1)

        upsample_layers = []
        for _ in range(int(math.log2(scale))):
            upsample_layers.extend([
                nn.Conv2d(channels, channels * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        self.upsample = nn.Sequential(*upsample_layers)

        self.conv_hr = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(channels, 3, 3, 1, 1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body = self.body(feat)
        body = self.conv_body(body)
        feat = feat + body
        out = self.upsample(feat)
        out = self.activation(self.conv_hr(out))
        out = self.conv_last(out)
        return torch.clamp(out, 0.0, 1.0)
