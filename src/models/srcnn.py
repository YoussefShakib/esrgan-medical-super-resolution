"""SRCNN (Super-Resolution Convolutional Neural Network) model."""

import torch
import torch.nn as nn


class SRCNN(nn.Module):
    """SRCNN model for super-resolution.
    
    Note: SRCNN expects upsampled input (unlike other models).
    Use bicubic upsampling before feeding to this model.
    """
    
    def __init__(self) -> None:
        """Initialize SRCNN with fixed architecture."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W) - should be upsampled already
            
        Returns:
            Super-resolved tensor (B, 3, H, W) with values in [0, 1]
        """
        return torch.clamp(self.net(x), 0.0, 1.0)
