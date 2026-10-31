"""EDSR model architecture."""

import math

import torch
import torch.nn as nn

from .common import UpsampleBlock


class EDSRResidualBlock(nn.Module):
    """EDSR residual block with residual scaling."""
    
    def __init__(self, channels: int = 64, residual_scale: float = 0.1) -> None:
        """
        Initialize EDSR residual block.
        
        Args:
            channels: Number of channels
            residual_scale: Residual scaling factor (0.1 recommended)
        """
        super().__init__()
        self.residual_scale = residual_scale
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return x + self.block(x) * self.residual_scale


class EDSR(nn.Module):
    """Enhanced Deep Residual Networks for Super-Resolution."""
    
    def __init__(
        self,
        scale: int = 4,
        num_blocks: int = 16,
        channels: int = 64,
        residual_scale: float = 0.1
    ) -> None:
        """
        Initialize EDSR model.
        
        Args:
            scale: Upsampling scale (2, 4, or 8)
            num_blocks: Number of residual blocks
            channels: Number of feature channels
            residual_scale: Residual scaling factor
            
        Raises:
            ValueError: If scale not in {2, 4, 8}
        """
        super().__init__()
        if scale not in {2, 4, 8}:
            raise ValueError("EDSR currently supports scale 2, 4, or 8")

        self.conv_first = nn.Conv2d(3, channels, 3, 1, 1)
        self.body = nn.Sequential(*[EDSRResidualBlock(channels, residual_scale) for _ in range(num_blocks)])
        self.conv_body = nn.Conv2d(channels, channels, 3, 1, 1)

        upsample_blocks = []
        for _ in range(int(math.log2(scale))):
            upsample_blocks.append(UpsampleBlock(channels, 2))
        self.upsample = nn.Sequential(*upsample_blocks)

        self.conv_last = nn.Conv2d(channels, 3, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W) with values in [0, 1]
            
        Returns:
            Super-resolved tensor (B, 3, H*scale, W*scale) with values in [0, 1]
        """
        first = self.conv_first(x)
        body = self.body(first)
        body = self.conv_body(body)
        out = first + body
        out = self.upsample(out)
        out = self.conv_last(out)
        return torch.clamp(out, 0.0, 1.0)
