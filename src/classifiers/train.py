"""Classifier training loop with early stopping and CSV logging.

Supports BACH (4-class histopathology) and BRISC2025 (4-class brain MRI).
Saves best checkpoint (by val accuracy) to runs/classifier_{dataset}_{arch}/.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import (
    build_bach_dataset,
    build_brisc_dataset,
    get_transforms,
)
from .models import build_classifier, count_trainable_params


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _topk_accuracy(outputs: torch.Tensor, targets: torch.Tensor, k: int = 3) -> float:
    with torch.no_grad():
        _, pred = outputs.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))
        return correct.any(dim=1).float().mean().item()


def _check_leakage(
    train_paths: List[Path],
    val_paths: List[Path],
    dataset_name: str,
) -> None:
    train_stems = {p.stem for p in train_paths}
    val_stems = {p.stem for p in val_paths}
    overlap = train_stems & val_stems
    if overlap:
        raise RuntimeError(
            f"[{dataset_name}] Data leakage: {len(overlap)} filename stems appear in "
            f"both train and val sets. Example: {next(iter(overlap))}"
        )


def _class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = Counter(labels)
    n = len(labels)
    weights = torch.tensor(
        [n / (num_classes * max(counts[i], 1)) for i in range(num_classes)],
        dtype=torch.float32,
    )
    return weights


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def _build_datasets(dataset_name: str, project_root: Path):
    if dataset_name == "bach":
        train_ds = build_bach_dataset(
            project_root, "train", get_transforms("bach", "train")
        )
        val_ds = build_bach_dataset(
            project_root, "val", get_transforms("bach", "val")
        )
    elif dataset_name == "brisc":
        train_ds = build_brisc_dataset(
            project_root, "train", get_transforms("brisc", "train")
        )
        val_ds = build_brisc_dataset(
            project_root, "val", get_transforms("brisc", "val")
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: bach, brisc")
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    epoch: int,
    val_acc: float,
    val_loss: float,
    arch: str,
    dataset_name: str,
    class_names: List[str],
) -> None:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "arch": arch,
            "dataset": dataset_name,
            "num_classes": len(class_names),
            "class_names": class_names,
            "model_state": model.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
        },
        ckpt_dir / "best.pth",
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_classifier(
    dataset_name: str,
    arch: str,
    project_root: Path,
    output_dir: Optional[Path] = None,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 3e-4,
    patience: int = 10,
    device: Optional[torch.device] = None,
    seed: int = 42,
    use_class_weights: Optional[bool] = None,
) -> Dict[str, Any]:
    """Train a classifier and return best-epoch metrics.

    Args:
        dataset_name: 'bach' or 'brisc'
        arch:         'resnet50' or 'efficientnet_b0'
        project_root: root of the esr_med project
        output_dir:   where to save checkpoints/logs (default: runs/classifier_{dataset}_{arch})
        epochs:       maximum training epochs
        batch_size:   images per batch
        lr:           initial learning rate for AdamW
        patience:     early-stopping patience (epochs without val_acc improvement)
        device:       torch.device (auto-detects CUDA/CPU if None)
        seed:         random seed
        use_class_weights: True to use class-weighted CE; None = auto (True for brisc)
    """
    _set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if output_dir is None:
        output_dir = project_root / "runs" / f"classifier_{dataset_name}_{arch}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------
    print(f"[{dataset_name}/{arch}] Loading datasets …")
    train_ds, val_ds = _build_datasets(dataset_name, project_root)
    class_names = train_ds.class_names
    num_classes = len(class_names)

    _check_leakage(train_ds.paths, val_ds.paths, dataset_name)

    print(
        f"  train={len(train_ds)}  val={len(val_ds)}  "
        f"classes={class_names}"
    )

    # Windows-safe: num_workers=0 avoids multiprocessing issues
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )

    # -------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------
    model = build_classifier(arch, num_classes).to(device)
    n_trainable = count_trainable_params(model)
    print(f"  trainable params: {n_trainable:,}")

    # -------------------------------------------------------------------
    # Loss — class-weighted CE for imbalanced datasets
    # -------------------------------------------------------------------
    if use_class_weights is None:
        use_class_weights = (dataset_name == "brisc")
    if use_class_weights:
        weights = _class_weights(train_ds.labels, num_classes).to(device)
        print(f"  class weights: {weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # -------------------------------------------------------------------
    # Optimiser + schedule: 5-epoch linear warmup then cosine decay
    # -------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4,
    )
    warmup_epochs = min(5, epochs // 5)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )

    # -------------------------------------------------------------------
    # CSV log
    # -------------------------------------------------------------------
    log_path = output_dir / "train_log.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow(
        ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_acc_top3", "lr"]
    )

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------
    best_val_acc = -1.0
    best_epoch = 0
    patience_counter = 0
    best_metrics: Dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, targets in tqdm(
            train_loader, desc=f"Epoch {epoch:>3}/{epochs} [train]", leave=False
        ):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == targets).sum().item()
            train_total += images.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Val ---
        model.eval()
        val_loss, val_correct, val_correct_top3, val_total = 0.0, 0, 0.0, 0
        with torch.no_grad():
            for images, targets in tqdm(
                val_loader, desc=f"Epoch {epoch:>3}/{epochs} [val]  ", leave=False
            ):
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == targets).sum().item()
                val_correct_top3 += _topk_accuracy(outputs, targets, k=min(3, num_classes)) * images.size(0)
                val_total += images.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_acc_top3 = val_correct_top3 / val_total
        current_lr = optimizer.param_groups[0]["lr"]

        # --- Log ---
        log_writer.writerow([
            epoch, f"{train_loss:.4f}", f"{train_acc:.4f}",
            f"{val_loss:.4f}", f"{val_acc:.4f}", f"{val_acc_top3:.4f}",
            f"{current_lr:.2e}",
        ])
        log_file.flush()

        print(
            f"  Epoch {epoch:>3}  train_loss={train_loss:.4f}  train_acc={train_acc:.3f}"
            f"  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  top3={val_acc_top3:.3f}"
            f"  lr={current_lr:.2e}"
        )

        scheduler.step()

        # --- Early stopping ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            _save_checkpoint(
                output_dir, model, epoch, val_acc, val_loss,
                arch, dataset_name, class_names,
            )
            best_metrics = {
                "epoch": epoch,
                "arch": arch,
                "dataset": dataset_name,
                "class_names": class_names,
                "val_acc": val_acc,
                "val_acc_top3": val_acc_top3,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "train_loss": train_loss,
            }
            print(f"    ✓ New best val_acc={val_acc:.4f} — checkpoint saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    log_file.close()

    # Persist best metrics as JSON
    metrics_path = output_dir / "best_metrics.json"
    metrics_path.write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    print(
        f"\n[{dataset_name}/{arch}] Done. Best val_acc={best_val_acc:.4f} at epoch {best_epoch}."
        f" Saved to {output_dir}"
    )
    return best_metrics
