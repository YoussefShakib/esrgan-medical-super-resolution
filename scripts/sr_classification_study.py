"""SR–Classification controlled study (Phase 3 — the 'killer experiment').

Classifies the SAME validation images at three quality levels:
  (a) Original HR  — ground-truth resolution
  (b) Degraded LR  — HR → bicubic ÷4 → bicubic ×4 (matched pipeline)
  (c) ESRGAN SR    — HR → bicubic ÷4 → ESRGAN ×4

This directly answers: does super-resolution help or hurt downstream
classification?  Results are stored regardless of direction.

Outputs
-------
  runs/sr_classification_study.json    — raw per-image predictions + summary
  figures/classifiers/sr_study_comparison.png — grouped bar chart

Usage
-----
    python scripts/sr_classification_study.py
    python scripts/sr_classification_study.py --dataset bach
    python scripts/sr_classification_study.py --skip-sr   # skip ESRGAN leg (no checkpoint needed)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# SR service (uses the existing tiled ESRGAN inference pipeline)
APPSERVICES_DIR = PROJECT_ROOT
if str(APPSERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(APPSERVICES_DIR))

from classifiers.models import build_classifier  # noqa: E402
from classifiers.dataset import (  # noqa: E402
    build_bach_dataset,
    build_brisc_dataset,
    get_transforms,
    BACH_CLASSES,
    BRISC_CLASSES,
)

try:
    from sklearn.metrics import f1_score, accuracy_score
except ImportError as e:
    raise ImportError("scikit-learn is required: pip install scikit-learn>=1.3.0") from e

SCALE = 4

# SR model registry for each dataset
SR_REGISTRY: Dict[str, Dict[str, str]] = {
    "bach": {
        "config": "configs/esrgan_bach_medical.yaml",
        "checkpoint": "runs/esrgan_bach_x4/checkpoints/best_ssim_generator.pth",
        "checkpoint_fallback": "runs/esrgan_bach_x4/checkpoints/best_generator.pth",
    },
    "brisc": {
        "config": "configs/esrgan_brain_mri.yaml",
        "checkpoint": "runs/esrgan_brain_mri_x4/checkpoints/best_ssim_generator.pth",
        "checkpoint_fallback": "runs/esrgan_brain_mri_x4/checkpoints/best_generator.pth",
    },
}

# Best classifier arch per dataset (try resnet50 first, then efficientnet_b0)
CLASSIFIER_ARCH_PRIORITY = ["resnet50", "efficientnet_b0"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_best_classifier(
    dataset_name: str, device: torch.device
) -> Tuple[Optional[torch.nn.Module], Optional[List[str]], Optional[str]]:
    """Load the best available classifier checkpoint for this dataset."""
    for arch in CLASSIFIER_ARCH_PRIORITY:
        ckpt_path = (
            PROJECT_ROOT / "runs" / f"classifier_{dataset_name}_{arch}" / "checkpoints" / "best.pth"
        )
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(str(ckpt_path), map_location=device)
        num_classes = ckpt["num_classes"]
        class_names = ckpt["class_names"]
        model = build_classifier(arch, num_classes).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"  Loaded classifier: {arch}  epoch={ckpt.get('epoch')}  val_acc={ckpt.get('val_acc', '?'):.4f}")
        return model, class_names, arch
    return None, None, None


def _load_sr_service(dataset_name: str, device: torch.device):
    """Load the ESRGAN SR service for this dataset. Returns None if not available."""
    reg = SR_REGISTRY[dataset_name]
    config_path = PROJECT_ROOT / reg["config"]
    ckpt_path = PROJECT_ROOT / reg["checkpoint"]
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / reg["checkpoint_fallback"]
    if not ckpt_path.exists() or not config_path.exists():
        return None
    try:
        from app_services.sr_service import load_sr_service
        svc = load_sr_service(str(config_path), str(ckpt_path))
        print(f"  Loaded SR service: {ckpt_path.name}")
        return svc
    except Exception as e:
        print(f"  WARNING: Could not load SR service: {e}")
        return None


def _bicubic_degraded(hr_pil: Image.Image, scale: int = SCALE) -> Image.Image:
    """HR → bicubic LR → bicubic back to HR size (the pseudo-LR condition)."""
    w, h = hr_pil.size
    w -= w % scale
    h -= h % scale
    hr_cropped = hr_pil.crop((0, 0, w, h))
    lr = hr_cropped.resize((w // scale, h // scale), Image.BICUBIC)
    return lr.resize((w, h), Image.BICUBIC)


def _classify_pil_batch(
    images_pil: List[Image.Image],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    """Classify a list of PIL images, return softmax probabilities [N, C]."""
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    all_probs = []
    for start in range(0, len(images_pil), batch_size):
        batch_imgs = images_pil[start : start + batch_size]
        tensors = torch.stack([val_tf(img.convert("RGB")) for img in batch_imgs]).to(device)
        with torch.no_grad():
            logits = model(tensors)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def _condition_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    preds = probs.argmax(axis=1)
    return {
        "top1_acc": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run_study(
    dataset_name: str,
    device: torch.device,
    skip_sr: bool = False,
    max_samples: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    print(f"\n{'='*60}")
    print(f"  SR–Classification Study: dataset={dataset_name}")
    print(f"{'='*60}")

    # Load classifier
    model, class_names, classifier_arch = _load_best_classifier(dataset_name, device)
    if model is None:
        print(f"  [SKIP] No trained classifier found for '{dataset_name}'. Train first.")
        return None

    # Load SR model (optional)
    sr_service = None if skip_sr else _load_sr_service(dataset_name, device)
    if not skip_sr and sr_service is None:
        print("  SR model not available — SR condition will be omitted.")

    # Load validation images (raw PIL, without classification transforms)
    print("  Loading validation images …")
    raw_val_ds = (
        build_bach_dataset(PROJECT_ROOT, "val", transform=None)
        if dataset_name == "bach"
        else build_brisc_dataset(PROJECT_ROOT, "val", transform=None)
    )
    labels = np.array(raw_val_ds.labels)

    if max_samples is not None and max_samples < len(raw_val_ds):
        idxs = np.random.choice(len(raw_val_ds), max_samples, replace=False)
        paths = [raw_val_ds.paths[i] for i in idxs]
        labels = labels[idxs]
    else:
        paths = raw_val_ds.paths

    hr_images = [Image.open(p).convert("RGB") for p in tqdm(paths, desc="  Loading HR images")]
    print(f"  Loaded {len(hr_images)} images.")

    results: Dict[str, Any] = {
        "dataset": dataset_name,
        "classifier_arch": classifier_arch,
        "class_names": class_names,
        "n_samples": len(hr_images),
        "conditions": {},
    }

    # -----------------------------------------------------------------------
    # Condition (a): Original HR
    # -----------------------------------------------------------------------
    print("  Running condition (a): Original HR …")
    probs_hr = _classify_pil_batch(hr_images, model, device)
    metrics_hr = _condition_metrics(probs_hr, labels)
    results["conditions"]["hr"] = {
        "label": "Original HR",
        "metrics": metrics_hr,
        "predictions": probs_hr.argmax(axis=1).tolist(),
    }
    print(f"    HR:       top1={metrics_hr['top1_acc']:.3f}  macro_f1={metrics_hr['macro_f1']:.3f}")

    # -----------------------------------------------------------------------
    # Condition (b): Degraded LR (bicubic ÷4 → ×4)
    # -----------------------------------------------------------------------
    print("  Running condition (b): Degraded LR (bicubic ÷4 → ×4) …")
    lr_images = [_bicubic_degraded(img) for img in tqdm(hr_images, desc="  Degrading", leave=False)]
    probs_lr = _classify_pil_batch(lr_images, model, device)
    metrics_lr = _condition_metrics(probs_lr, labels)
    results["conditions"]["lr_bicubic"] = {
        "label": "Degraded LR (bicubic ÷4 → ×4)",
        "metrics": metrics_lr,
        "predictions": probs_lr.argmax(axis=1).tolist(),
    }
    print(f"    LR bicubic: top1={metrics_lr['top1_acc']:.3f}  macro_f1={metrics_lr['macro_f1']:.3f}")

    # -----------------------------------------------------------------------
    # Condition (c): ESRGAN SR
    # -----------------------------------------------------------------------
    if sr_service is not None:
        print("  Running condition (c): ESRGAN SR (HR → LR → ESRGAN) …")
        sr_images = []
        for img in tqdm(hr_images, desc="  ESRGAN SR", leave=False):
            w, h = img.size
            w -= w % SCALE
            h -= h % SCALE
            hr_cropped = img.crop((0, 0, w, h))
            lr = hr_cropped.resize((w // SCALE, h // SCALE), Image.BICUBIC)
            sr = sr_service.enhance_pil_image(lr, use_tiling=True)
            sr_images.append(sr)

        probs_sr = _classify_pil_batch(sr_images, model, device)
        metrics_sr = _condition_metrics(probs_sr, labels)
        results["conditions"]["esrgan_sr"] = {
            "label": "ESRGAN SR (HR → LR → ESRGAN ×4)",
            "metrics": metrics_sr,
            "predictions": probs_sr.argmax(axis=1).tolist(),
        }
        print(f"    ESRGAN SR:  top1={metrics_sr['top1_acc']:.3f}  macro_f1={metrics_sr['macro_f1']:.3f}")

        # Interpret the finding honestly
        sr_vs_lr = metrics_sr["top1_acc"] - metrics_lr["top1_acc"]
        hr_vs_sr = metrics_hr["top1_acc"] - metrics_sr["top1_acc"]
        if sr_vs_lr > 0.02:
            finding = "SR improves classification accuracy over degraded LR."
        elif sr_vs_lr < -0.02:
            finding = "SR slightly degrades classification accuracy vs degraded LR."
        else:
            finding = "SR has negligible effect on classification accuracy vs degraded LR."
        if hr_vs_sr > 0.05:
            finding += " However, HR remains substantially better than SR."
        results["finding"] = finding
        print(f"  Finding: {finding}")
    else:
        results["finding"] = "ESRGAN SR condition skipped (no checkpoint available)."

    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot_comparison(study_results: Dict[str, Any], out_path: Path) -> None:
    """Grouped bar chart: top-1 accuracy + macro F1 per condition per dataset."""
    datasets_available = list(study_results.keys())
    metrics_to_show = ["top1_acc", "macro_f1"]
    metric_labels = {"top1_acc": "Top-1 Accuracy", "macro_f1": "Macro F1"}

    n_datasets = len(datasets_available)
    fig, axes = plt.subplots(1, n_datasets, figsize=(7 * n_datasets, 5), squeeze=False)

    for col, ds_name in enumerate(datasets_available):
        ax = axes[0][col]
        ds_result = study_results[ds_name]
        conditions = ds_result.get("conditions", {})
        cond_names = list(conditions.keys())
        cond_labels = [conditions[c]["label"] for c in cond_names]

        x = np.arange(len(cond_names))
        width = 0.35
        colors = ["#4FC3B8", "#FF8C69"]

        for m_idx, metric in enumerate(metrics_to_show):
            vals = [conditions[c]["metrics"][metric] for c in cond_names]
            bars = ax.bar(x + m_idx * width, vals, width, label=metric_labels[metric], color=colors[m_idx], alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8,
                )

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(cond_labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Score")
        ax.set_title(
            f"SR–Classification Study: {ds_name.upper()}\n"
            f"({ds_result.get('classifier_arch', '')}  n={ds_result.get('n_samples', '?')})"
        )
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Annotate the finding
        finding = ds_result.get("finding", "")
        ax.text(
            0.5, -0.22, finding, transform=ax.transAxes,
            ha="center", va="top", fontsize=8, style="italic", wrap=True,
        )

    fig.suptitle("Does super-resolution help downstream classification?", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Comparison plot saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SR–Classification controlled study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset", choices=["bach", "brisc"],
        help="Run study for a single dataset (default: both)"
    )
    p.add_argument(
        "--skip-sr", action="store_true",
        help="Skip the ESRGAN SR condition (useful when SR checkpoints are unavailable)"
    )
    p.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of val images per dataset (for quick testing)"
    )
    p.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "runs" / "sr_classification_study.json",
        help="Path for the JSON results file",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    datasets = [args.dataset] if args.dataset else ["bach", "brisc"]
    study_results: Dict[str, Any] = {}

    for ds in datasets:
        result = run_study(
            ds, device,
            skip_sr=args.skip_sr,
            max_samples=args.max_samples,
        )
        if result is not None:
            study_results[ds] = result

    if not study_results:
        print("\nNo results to save (train classifiers first).")
        return

    # Save JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(study_results, indent=2), encoding="utf-8")
    print(f"\nStudy results saved to {args.output}")

    # Comparison plot
    out_plot = PROJECT_ROOT / "figures" / "classifiers" / "sr_study_comparison.png"
    _plot_comparison(study_results, out_plot)

    # Print summary table
    print("\n" + "=" * 60)
    print("Study Summary")
    print("=" * 60)
    for ds, res in study_results.items():
        print(f"\n  {ds.upper()}  (arch={res.get('classifier_arch', '?')})")
        for cond_key, cond_data in res.get("conditions", {}).items():
            m = cond_data["metrics"]
            print(
                f"    {cond_data['label']:40s}"
                f"  top1={m['top1_acc']:.3f}  macro_f1={m['macro_f1']:.3f}"
            )
        print(f"  Finding: {res.get('finding', '—')}")


if __name__ == "__main__":
    main()
