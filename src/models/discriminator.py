\
import torch
import torch.nn as nn


def block(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


class Discriminator(nn.Module):
    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(3, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            block(base_channels, base_channels, 2),
            block(base_channels, base_channels * 2, 1),
            block(base_channels * 2, base_channels * 2, 2),
            block(base_channels * 2, base_channels * 4, 1),
            block(base_channels * 4, base_channels * 4, 2),
            block(base_channels * 4, base_channels * 8, 1),
            block(base_channels * 8, base_channels * 8, 2),
        ]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 8, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)
