"""Classifier service for the Streamlit Research Prediction tab.

Wraps a trained classifier checkpoint and exposes:
  - predict(pil_image) -> PredictionResult
  - gradcam(pil_image, class_idx) -> np.ndarray (heatmap, same size as input)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from classifiers.models import build_classifier  # noqa: E402

_GRADCAM_OK = False
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    _GRADCAM_OK = True
except ImportError:
    pass

_VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class PredictionResult:
    predicted_class: str
    confidence: float
    probabilities: List[float]
    class_names: List[str]


class ClassifierService:
    """Wraps a trained classifier for inference in the Streamlit app."""

    def __init__(
        self,
        model: torch.nn.Module,
        class_names: List[str],
        arch: str,
        dataset_name: str,
        checkpoint_epoch: int,
        val_acc: float,
        device: torch.device,
    ) -> None:
        self.model = model
        self.class_names = class_names
        self.arch = arch
        self.dataset_name = dataset_name
        self.checkpoint_epoch = checkpoint_epoch
        self.val_acc = val_acc
        self.device = device
        self._cam: Optional[object] = None

    def predict(self, image: Image.Image) -> PredictionResult:
        tensor = _VAL_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        top_idx = int(probs.argmax())
        return PredictionResult(
            predicted_class=self.class_names[top_idx],
            confidence=float(probs[top_idx]),
            probabilities=probs.tolist(),
            class_names=self.class_names,
        )

    def gradcam_overlay(self, image: Image.Image, class_idx: Optional[int] = None) -> Optional[np.ndarray]:
        """Return a Grad-CAM overlay as an RGB numpy array (same HxW as `image` after 224-crop).

        Returns None if pytorch-grad-cam is not installed.
        """
        if not _GRADCAM_OK:
            return None

        if self._cam is None:
            target_layer = self._get_target_layer()
            self._cam = GradCAM(model=self.model, target_layers=[target_layer])

        tensor = _VAL_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self.device)
        # Use predicted class if none specified
        if class_idx is None:
            with torch.no_grad():
                class_idx = int(self.model(tensor).argmax(1).item())

        targets = [ClassifierOutputTarget(class_idx)]
        grayscale_cam = self._cam(input_tensor=tensor, targets=targets)[0]

        # Denormalize for display
        img_array = tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        img_array = np.clip(img_array * _IMAGENET_STD + _IMAGENET_MEAN, 0, 1).astype(np.float32)
        overlay = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
        return overlay  # uint8 HxWx3

    def _get_target_layer(self) -> torch.nn.Module:
        if self.arch == "resnet50":
            return self.model.layer4[-1].conv3
        elif self.arch == "efficientnet_b0":
            return self.model.blocks[-1][-1].conv_pw
        raise ValueError(f"No Grad-CAM target layer for arch '{self.arch}'")

    @property
    def label(self) -> str:
        return f"{self.dataset_name.upper()} {self.arch} (ep {self.checkpoint_epoch}, val_acc={self.val_acc:.3f})"


def load_classifier_service(
    dataset_name: str,
    arch: Optional[str] = None,
) -> Optional[ClassifierService]:
    """Load the best available classifier checkpoint for the given dataset.

    Tries arches in priority order if `arch` is None.
    Returns None if no trained checkpoint exists yet.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch_priority = [arch] if arch else ["resnet50", "efficientnet_b0"]

    for try_arch in arch_priority:
        ckpt_path = (
            PROJECT_ROOT / "runs" / f"classifier_{dataset_name}_{try_arch}"
            / "checkpoints" / "best.pth"
        )
        if not ckpt_path.exists():
            continue
        try:
            ckpt = torch.load(str(ckpt_path), map_location=device)
            num_classes = ckpt["num_classes"]
            class_names = ckpt["class_names"]
            model = build_classifier(try_arch, num_classes).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            return ClassifierService(
                model=model,
                class_names=class_names,
                arch=try_arch,
                dataset_name=dataset_name,
                checkpoint_epoch=ckpt.get("epoch", -1),
                val_acc=ckpt.get("val_acc", 0.0),
                device=device,
            )
        except Exception as e:
            print(f"WARNING: Could not load classifier {try_arch} for {dataset_name}: {e}")
            continue

    return None
