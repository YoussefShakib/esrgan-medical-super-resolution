"""Classification datasets for BACH histopathology and BRISC2025 brain MRI.

BACH layout  : datasets/train_HR_stratified/ and val_HR_stratified/
               filenames encode class: Normal_n001.tif, Benign_b001.tif, etc.

BRISC layout : datasets/BRAIN_MRI/raw/{class}/train/ and test/
               class folders: glioma, meningioma, no_tumor, pituitary
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

BACH_CLASSES: List[str] = ["Normal", "Benign", "InSitu", "Invasive"]
BRISC_CLASSES: List[str] = ["glioma", "meningioma", "no_tumor", "pituitary"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


class ClassificationDataset(Dataset):
    def __init__(
        self,
        paths: List[Path],
        labels: List[int],
        class_names: List[str],
        transform: Optional[Callable] = None,
    ) -> None:
        if len(paths) != len(labels):
            raise ValueError("paths and labels must have the same length")
        self.paths = paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


# ---------------------------------------------------------------------------
# BACH builders
# ---------------------------------------------------------------------------

def _parse_bach_class(stem: str) -> Optional[str]:
    """Extract class from BACH filename stem (e.g. 'Normal_n001' -> 'Normal')."""
    for cls in BACH_CLASSES:
        if stem.lower().startswith(cls.lower() + "_"):
            return cls
    return None


def build_bach_dataset(
    project_root: Path,
    split: str,
    transform: Optional[Callable] = None,
) -> ClassificationDataset:
    """Load BACH from the stratified HR split directories.

    split: 'train' -> train_HR_stratified/  |  'val' -> val_HR_stratified/
    """
    split_dirs = {
        "train": project_root / "datasets" / "train_HR_stratified",
        "val": project_root / "datasets" / "val_HR_stratified",
    }
    if split not in split_dirs:
        raise ValueError(f"split must be 'train' or 'val', got '{split}'")
    root = split_dirs[split]
    if not root.exists():
        raise FileNotFoundError(f"BACH split directory not found: {root}")

    label_map = {cls: i for i, cls in enumerate(BACH_CLASSES)}
    paths, labels = [], []
    for p in sorted(root.iterdir()):
        if p.suffix.lower() not in _IMG_EXTS:
            continue
        cls = _parse_bach_class(p.stem)
        if cls is None:
            continue
        paths.append(p)
        labels.append(label_map[cls])

    if not paths:
        raise RuntimeError(f"No images found in {root}")
    return ClassificationDataset(paths, labels, BACH_CLASSES, transform)


# ---------------------------------------------------------------------------
# BRISC2025 builders
# ---------------------------------------------------------------------------

def build_brisc_dataset(
    project_root: Path,
    split: str,
    transform: Optional[Callable] = None,
) -> ClassificationDataset:
    """Load BRISC2025 from datasets/BRAIN_MRI/raw/{class}/train|test/.

    split: 'train' -> raw/{class}/train/  |  'val' or 'test' -> raw/{class}/test/
    """
    raw_root = project_root / "datasets" / "BRAIN_MRI" / "raw"
    if not raw_root.exists():
        raise FileNotFoundError(f"BRISC2025 raw directory not found: {raw_root}")

    folder_name = "train" if split == "train" else "test"
    label_map = {cls: i for i, cls in enumerate(BRISC_CLASSES)}
    paths, labels = [], []
    for cls in BRISC_CLASSES:
        cls_dir = raw_root / cls / folder_name
        if not cls_dir.exists():
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in _IMG_EXTS:
                paths.append(p)
                labels.append(label_map[cls])

    if not paths:
        raise RuntimeError(
            f"No images found under {raw_root} for split='{split}'"
        )
    return ClassificationDataset(paths, labels, BRISC_CLASSES, transform)


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

def get_transforms(domain: str, split: str) -> Callable:
    """Return the augmentation pipeline for (domain, split).

    domain: 'bach' | 'brisc'
    split:  'train' | 'val'
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split != "train":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    if domain == "bach":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])

    if domain == "brisc":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            normalize,
        ])

    raise ValueError(f"Unknown domain '{domain}'. Choose from: bach, brisc")
