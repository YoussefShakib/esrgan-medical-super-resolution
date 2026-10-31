\
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG19_Weights


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((prediction - target) ** 2 + self.epsilon ** 2))


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_index: int = 35) -> None:
        super().__init__()
        weights = VGG19_Weights.IMAGENET1K_V1
        vgg_features = models.vgg19(weights=weights).features[:layer_index].eval()
        for param in vgg_features.parameters():
            param.requires_grad = False
        self.features = vgg_features
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.features(x)
