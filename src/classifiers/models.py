"""Classifier model factories (ResNet50, EfficientNet-B0) via timm.

Unfreezing strategy: freeze the full backbone, then selectively unfreeze
the last residual/MBConv block and the classification head.  This avoids
catastrophic forgetting while letting the model adapt its high-level
features to medical-image classes.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError("timm is required: pip install timm>=0.9.0") from e


def _freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def build_resnet50_classifier(num_classes: int) -> nn.Module:
    model = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)
    _freeze_all(model)
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True
    return model


def build_efficientnet_b0_classifier(num_classes: int) -> nn.Module:
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    _freeze_all(model)
    # Unfreeze last MBConv stage, head conv, batch norm, and classifier
    for p in model.blocks[-1].parameters():
        p.requires_grad = True
    for name, p in model.named_parameters():
        if any(k in name for k in ("conv_head", "bn2", "classifier")):
            p.requires_grad = True
    return model


ARCH_BUILDERS: Dict[str, Callable[[int], nn.Module]] = {
    "resnet50": build_resnet50_classifier,
    "efficientnet_b0": build_efficientnet_b0_classifier,
}


def build_classifier(arch: str, num_classes: int) -> nn.Module:
    if arch not in ARCH_BUILDERS:
        raise ValueError(f"Unknown arch '{arch}'. Choose from: {list(ARCH_BUILDERS)}")
    return ARCH_BUILDERS[arch](num_classes)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
