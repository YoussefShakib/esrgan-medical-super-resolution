"""Evaluation script for trained classifiers.

Generates per-checkpoint:
  - Confusion matrix (normalized + raw counts)
  - Per-class precision / recall / F1
  - ROC curves with per-class AUC (one-vs-rest)
  - Top-1 and top-3 accuracy
  - Calibration plot (reliability diagram + Expected Calibration Error)
  - Grad-CAM heatmaps: 5 correct + 5 misclassified images per class

All figures saved to figures/classifiers/{dataset}_{arch}/
Summary saved to runs/classifier_results.json

Usage
-----
# Evaluate a single checkpoint:
    python scripts/eval_classifiers.py --dataset bach --arch resnet50

# Evaluate all trained checkpoints:
    python scripts/eval_classifiers.py --all

# Only generate Grad-CAM (skips other metrics):
    python scripts/eval_classifiers.py --dataset brisc --arch efficientnet_b0 --gradcam-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from classifiers.models import build_classifier  # noqa: E402
from classifiers.dataset import (  # noqa: E402
    build_bach_dataset,
    build_brisc_dataset,
    get_transforms,
)

try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.calibration import calibration_curve
except ImportError as e:
    raise ImportError("scikit-learn is required: pip install scikit-learn>=1.3.0") from e

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    _GRADCAM_AVAILABLE = True
except ImportError:
    _GRADCAM_AVAILABLE = False
    print("WARNING: pytorch-grad-cam not available — Grad-CAM will be skipped.")

DATASETS = ["bach", "brisc"]
ARCHS = ["resnet50", "efficientnet_b0"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(ckpt_path: Path, device: torch.device) -> Dict:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(str(ckpt_path), map_location=device)


def _get_target_layer(model: torch.nn.Module, arch: str) -> torch.nn.Module:
    """Return the last conv layer used for Grad-CAM."""
    if arch == "resnet50":
        return model.layer4[-1].conv3
    elif arch == "efficientnet_b0":
        return model.blocks[-1][-1].conv_pw
    raise ValueError(f"No target layer defined for arch '{arch}'")


def _build_val_dataset(dataset_name: str):
    if dataset_name == "bach":
        return build_bach_dataset(
            PROJECT_ROOT, "val", get_transforms("bach", "val")
        )
    return build_brisc_dataset(
        PROJECT_ROOT, "val", get_transforms("brisc", "val")
    )


def _topk_acc(outputs: np.ndarray, targets: np.ndarray, k: int) -> float:
    topk = np.argsort(outputs, axis=1)[:, -k:]
    correct = np.any(topk == targets.reshape(-1, 1), axis=1)
    return float(correct.mean())


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += mask.sum() * abs(acc - conf)
    return float(ece / len(labels))


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, subtitle in zip(
        axes,
        [cm.astype(float) / cm.sum(axis=1, keepdims=True), cm],
        [".2f", "d"],
        ["Normalized", "Raw counts"],
    ):
        im = ax.imshow(data, interpolation="nearest", cmap="Blues")
        ax.set(
            xticks=range(len(class_names)),
            yticks=range(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Predicted",
            ylabel="True",
            title=f"{title} — {subtitle}",
        )
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        thresh = data.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(
                    j, i, format(data[i, j], fmt),
                    ha="center", va="center",
                    color="white" if data[i, j] > thresh else "black",
                    fontsize=9,
                )
        plt.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_roc(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    out_path: Path,
    title: str,
) -> Dict[str, float]:
    n = len(class_names)
    labels_onehot = np.eye(n)[labels]
    fig, ax = plt.subplots(figsize=(7, 6))
    aucs = {}
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(labels_onehot[:, i], probs[:, i])
        auc = roc_auc_score(labels_onehot[:, i], probs[:, i])
        aucs[cls] = float(auc)
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return aucs


def _plot_calibration(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    ece: float,
    out_path: Path,
    title: str,
) -> None:
    n = len(class_names)
    labels_onehot = np.eye(n)[labels]
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, cls in enumerate(class_names):
        fraction_pos, mean_pred = calibration_curve(
            labels_onehot[:, i], probs[:, i], n_bins=10, strategy="uniform"
        )
        ax.plot(mean_pred, fraction_pos, marker="o", markersize=4, label=cls)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Fraction of positives",
        title=f"{title}\nECE = {ece:.4f}",
        xlim=[0, 1], ylim=[0, 1],
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Grad-CAM generation
# ---------------------------------------------------------------------------

def _generate_gradcam(
    model: torch.nn.Module,
    arch: str,
    dataset_name: str,
    all_probs: np.ndarray,
    all_labels: np.ndarray,
    val_dataset,
    out_dir: Path,
    device: torch.device,
    n_per_class: int = 5,
) -> None:
    if not _GRADCAM_AVAILABLE:
        print("  Grad-CAM skipped (pytorch-grad-cam not installed).")
        return

    target_layer = _get_target_layer(model, arch)
    cam = GradCAM(model=model, target_layers=[target_layer])

    preds = all_probs.argmax(axis=1)
    correct_mask = preds == all_labels
    class_names = val_dataset.class_names

    for cls_idx, cls_name in enumerate(class_names):
        cls_correct = np.where(correct_mask & (all_labels == cls_idx))[0]
        cls_wrong = np.where(~correct_mask & (all_labels == cls_idx))[0]

        for subset_name, idxs in [("correct", cls_correct), ("misclassified", cls_wrong)]:
            chosen = idxs[:n_per_class]
            if len(chosen) == 0:
                continue
            fig, axes = plt.subplots(1, len(chosen), figsize=(4 * len(chosen), 4))
            if len(chosen) == 1:
                axes = [axes]
            for ax, sample_idx in zip(axes, chosen):
                img_tensor, _ = val_dataset[int(sample_idx)]
                input_t = img_tensor.unsqueeze(0).to(device)
                target = [ClassifierOutputTarget(cls_idx)]
                grayscale_cam = cam(input_tensor=input_t, targets=target)[0]
                # Denormalize for display
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                rgb = img_tensor.permute(1, 2, 0).numpy() * std + mean
                rgb = np.clip(rgb, 0, 1).astype(np.float32)
                visualization = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
                pred_cls = class_names[preds[sample_idx]]
                ax.imshow(visualization)
                ax.set_title(f"True:{cls_name}\nPred:{pred_cls}", fontsize=8)
                ax.axis("off")
            fig.suptitle(
                f"{dataset_name.upper()} Grad-CAM — class={cls_name} — {subset_name}",
                fontsize=10,
            )
            fig.tight_layout()
            save_path = out_dir / f"gradcam_{cls_name}_{subset_name}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

    cam.__del__() if hasattr(cam, "__del__") else None


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    dataset_name: str,
    arch: str,
    device: torch.device,
    gradcam_only: bool = False,
) -> Optional[Dict[str, Any]]:
    ckpt_dir = PROJECT_ROOT / "runs" / f"classifier_{dataset_name}_{arch}"
    ckpt_path = ckpt_dir / "checkpoints" / "best.pth"

    if not ckpt_path.exists():
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        return None

    print(f"\n  Loading checkpoint: {ckpt_path}")
    ckpt = _load_checkpoint(ckpt_path, device)
    class_names: List[str] = ckpt["class_names"]
    num_classes = len(class_names)

    model = build_classifier(arch, num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_ds = _build_val_dataset(dataset_name)
    val_loader = DataLoader(
        val_ds, batch_size=16, shuffle=False, num_workers=0
    )

    # --- Inference ---
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="  Inference", leave=False):
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(targets.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    preds = all_probs.argmax(axis=1)

    fig_dir = PROJECT_ROOT / "figures" / "classifiers" / f"{dataset_name}_{arch}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    title = f"{dataset_name.upper()} — {arch}"

    if not gradcam_only:
        # Confusion matrix
        cm = confusion_matrix(all_labels, preds)
        _plot_confusion_matrix(cm, class_names, fig_dir / "confusion_matrix.png", title)

        # Per-class metrics
        report = classification_report(
            all_labels, preds, target_names=class_names, output_dict=True
        )

        # ROC + AUC
        aucs = _plot_roc(all_probs, all_labels, class_names, fig_dir / "roc_curves.png", title)

        # Accuracy
        top1 = _topk_acc(all_probs, all_labels, k=1)
        top3 = _topk_acc(all_probs, all_labels, k=min(3, num_classes))

        # Calibration
        ece = _ece(all_probs, all_labels)
        _plot_calibration(all_probs, all_labels, class_names, ece, fig_dir / "calibration.png", title)

        # Data-leakage flag (>95% accuracy triggers warning)
        if top1 > 0.95:
            print(
                f"  WARNING: val_acc={top1:.3f} > 0.95 — investigate for data leakage "
                f"or distribution mismatch before reporting this result."
            )

        results = {
            "dataset": dataset_name,
            "arch": arch,
            "checkpoint_epoch": ckpt.get("epoch"),
            "class_names": class_names,
            "top1_acc": top1,
            "top3_acc": top3,
            "ece": ece,
            "auc_per_class": aucs,
            "macro_auc": float(np.mean(list(aucs.values()))),
            "per_class_metrics": {
                cls: {
                    "precision": report[cls]["precision"],
                    "recall": report[cls]["recall"],
                    "f1": report[cls]["f1-score"],
                    "support": report[cls]["support"],
                }
                for cls in class_names
            },
            "macro_avg": report["macro avg"],
            "weighted_avg": report["weighted avg"],
        }
        print(
            f"  top1={top1:.3f}  top3={top3:.3f}  ECE={ece:.4f}"
            f"  macro_AUC={results['macro_auc']:.3f}"
        )

    # Grad-CAM
    _generate_gradcam(
        model, arch, dataset_name, all_probs, all_labels, val_ds,
        fig_dir / f"gradcam_{dataset_name}_{arch}",
        device,
    )

    if gradcam_only:
        return None

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate trained BACH / BRISC2025 classifiers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", choices=DATASETS, help="Dataset")
    p.add_argument("--arch", choices=ARCHS, help="Architecture")
    p.add_argument("--all", action="store_true", help="Evaluate all trained checkpoints")
    p.add_argument(
        "--gradcam-only", action="store_true",
        help="Only regenerate Grad-CAM visualizations"
    )
    p.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "runs" / "classifier_results.json",
        help="Path for the combined JSON results file",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.all:
        combos = [(d, a) for d in DATASETS for a in ARCHS]
    elif args.dataset and args.arch:
        combos = [(args.dataset, args.arch)]
    else:
        print("ERROR: specify --dataset and --arch, or use --all")
        sys.exit(1)

    all_results: Dict[str, Any] = {}
    if args.output.exists():
        try:
            all_results = json.loads(args.output.read_text(encoding="utf-8"))
        except Exception:
            pass

    for dataset_name, arch in combos:
        print(f"\n{'='*60}")
        print(f"  Evaluating: dataset={dataset_name}  arch={arch}")
        print(f"{'='*60}")
        result = evaluate_checkpoint(
            dataset_name, arch, device, gradcam_only=args.gradcam_only
        )
        if result is not None:
            all_results[f"{dataset_name}_{arch}"] = result

    if not args.gradcam_only and all_results:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\nResults saved to {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
