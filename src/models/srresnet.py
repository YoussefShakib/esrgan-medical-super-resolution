\
import math
import torch
import torch.nn as nn

from .common import ResidualBlock, UpsampleBlock


class SRResNet(nn.Module):
    def __init__(self, scale: int = 4, num_blocks: int = 16, channels: int = 64) -> None:
        super().__init__()
        if scale not in {2, 4, 8}:
            raise ValueError("SRResNet currently supports scale 2, 4, or 8")

        self.conv_first = nn.Sequential(
            nn.Conv2d(3, channels, 9, 1, 4),
            nn.PReLU(),
        )

        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        self.conv_body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

        upsample_blocks = []
        for _ in range(int(math.log2(scale))):
            upsample_blocks.append(UpsampleBlock(channels, 2))
        self.upsample = nn.Sequential(*upsample_blocks)

        self.conv_last = nn.Conv2d(channels, 3, 9, 1, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.conv_first(x)
        body = self.body(first)
        body = self.conv_body(body)
        out = first + body
        out = self.upsample(out)
        out = self.conv_last(out)
        return torch.clamp(out, 0.0, 1.0)
