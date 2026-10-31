"""Train EDSR and SRResNet on the Brain MRI dataset.

Produces runs/edsr_brain_mri_x4 and runs/srresnet_brain_mri_x4
with the same 15-epoch budget as the existing ESRGAN MRI run so that
all four methods can be compared fairly.

Usage
-----
    python scripts/train_mri_baselines.py [--epochs N] [--skip-edsr] [--skip-srresnet]
    python scripts/train_mri_baselines.py --eval-only   (skip training, re-evaluate)
    python scripts/train_mri_baselines.py --compute-lpips

CPU runtime estimate (no GPU):
    EDSR    15 epochs x 1275 batches ~ 8-20 hours
    SRResNet 15 epochs x 1275 batches ~ 4-10 hours
  Total: ~12-30 hours.  Consider running with a GPU or overnight.

After training completes, evaluation results and a comparison figure
are written to runs/mri_evaluation.json and runs/mri_comparison.png.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.core.device import DeviceManager
from src.dataset import SuperResolutionDataset
from src.metrics import batch_metrics
from src.models import build_model
from src.utils import save_json

MRI_CONFIGS = {
    "edsr":     "configs/edsr_brain_mri.yaml",
    "srresnet": "configs/srresnet_brain_mri.yaml",
}
MRI_ESRGAN_BEST = "runs/esrgan_brain_mri_x4/checkpoints/best_generator.pth"
MRI_ESRGAN_BEST_FALLBACK = "runs/esrgan_brain_mri_x4/checkpoints/best_ssim_generator.pth"
MRI_ESRGAN_CONFIG = "configs/esrgan_brain_mri.yaml"

FAIRNESS_EPOCHS = 15  # matches the existing ESRGAN MRI run


def _best_checkpoint(config_path: str) -> Optional[Path]:
    cfg = load_config(config_path)
    exp_dir = Path(cfg.save_dir) / cfg.experiment_name / "checkpoints"
    for name in ("best.pth", "best_generator.pth"):
        p = exp_dir / name
        if p.exists():
            return p
    return None


def _run_training(config_path: str, epochs: int) -> None:
    """Launch train.py with epoch override via a patched config in memory."""
    import yaml, tempfile, os

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["training"]["epochs"] = epochs

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        yaml.dump(cfg, tmp)
        tmp_path = tmp.name

    try:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "train.py"),
            "--config", tmp_path,
        ]
        print(f"\n{'='*60}")
        print(f"Training {config_path} for {epochs} epochs")
        print(f"{'='*60}")
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            raise RuntimeError(f"Training failed: {config_path}")
    finally:
        os.unlink(tmp_path)


def _evaluate_checkpoint(
    config_path: str,
    checkpoint_path: str,
    model_name: str,
    compute_lpips: bool,
) -> Dict[str, float]:
    cfg = load_config(config_path)
    device = DeviceManager.setup(cfg.model_dump())

    dataset = SuperResolutionDataset(
        hr_dir=cfg.dataset.val_hr_dir,
        scale=cfg.scale,
        patch_size=cfg.dataset.patch_size,
        is_train=False,
        srcnn_mode=False,
        degradation_config={
            **cfg.dataset.degradation,
            "eval_crop_size": cfg.dataset.eval_crop_size,
        },
        image_mode=cfg.dataset.image_mode,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(model_name, cfg.scale).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    all_m = []
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            all_m.append(batch_metrics(sr, hr, compute_lpips=compute_lpips))

    n = len(all_m)
    result: Dict[str, float] = {
        "psnr": float(sum(m["psnr"] for m in all_m) / n),
        "ssim": float(sum(m["ssim"] for m in all_m) / n),
        "num_images": float(n),
    }
    if compute_lpips and all("lpips" in m for m in all_m):
        result["lpips"] = float(sum(m["lpips"] for m in all_m) / n)
    DeviceManager.cleanup()
    return result


def _eval_bicubic_mri(config_path: str, compute_lpips: bool) -> Dict[str, float]:
    import torchvision.transforms.functional as TF
    cfg = load_config(config_path)
    device = DeviceManager.setup(cfg.model_dump())

    dataset = SuperResolutionDataset(
        hr_dir=cfg.dataset.val_hr_dir,
        scale=cfg.scale,
        patch_size=cfg.dataset.patch_size,
        is_train=False,
        srcnn_mode=False,
        degradation_config={
            **cfg.dataset.degradation,
            "eval_crop_size": cfg.dataset.eval_crop_size,
        },
        image_mode=cfg.dataset.image_mode,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    all_m = []
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            h, w = lr.shape[-2], lr.shape[-1]
            bic = TF.resize(
                lr,
                size=[h * cfg.scale, w * cfg.scale],
                interpolation=TF.InterpolationMode.BICUBIC,
                antialias=True,
            ).clamp(0, 1)
            all_m.append(batch_metrics(bic, hr, compute_lpips=compute_lpips))

    n = len(all_m)
    result: Dict[str, float] = {
        "psnr": float(sum(m["psnr"] for m in all_m) / n),
        "ssim": float(sum(m["ssim"] for m in all_m) / n),
        "num_images": float(n),
    }
    if compute_lpips and all("lpips" in m for m in all_m):
        result["lpips"] = float(sum(m["lpips"] for m in all_m) / n)
    DeviceManager.cleanup()
    return result


def _build_comparison_figure(results: Dict[str, Dict[str, float]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    methods = list(results.keys())
    psnrs = [results[m].get("psnr", 0) for m in methods]
    ssims = [results[m].get("ssim", 0) for m in methods]
    has_lpips = all("lpips" in results[m] for m in methods)
    lpips = [results[m].get("lpips", 0) for m in methods] if has_lpips else []

    n_cols = 3 if has_lpips else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 4))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"][:len(methods)]

    axes[0].bar(methods, psnrs, color=colors)
    axes[0].set_title("PSNR (dB) — higher is better")
    axes[0].set_ylabel("dB")
    axes[0].set_ylim(min(psnrs) - 1, max(psnrs) + 1)
    for i, v in enumerate(psnrs):
        axes[0].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)

    axes[1].bar(methods, ssims, color=colors)
    axes[1].set_title("SSIM — higher is better")
    axes[1].set_ylim(min(ssims) - 0.02, max(ssims) + 0.02)
    for i, v in enumerate(ssims):
        axes[1].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)

    if has_lpips:
        axes[2].bar(methods, lpips, color=colors)
        axes[2].set_title("LPIPS — lower is better")
        axes[2].set_ylim(0, max(lpips) + 0.05)
        for i, v in enumerate(lpips):
            axes[2].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)

    plt.suptitle("Brain MRI SR: method comparison", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison figure saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train MRI baselines (EDSR + SRResNet)")
    parser.add_argument("--epochs", type=int, default=FAIRNESS_EPOCHS)
    parser.add_argument("--skip-edsr", action="store_true")
    parser.add_argument("--skip-srresnet", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help="Skip training")
    parser.add_argument("--compute-lpips", action="store_true")
    args = parser.parse_args()

    # Training
    if not args.eval_only:
        if not args.skip_edsr:
            _run_training(MRI_CONFIGS["edsr"], epochs=args.epochs)
        if not args.skip_srresnet:
            _run_training(MRI_CONFIGS["srresnet"], epochs=args.epochs)

    # Evaluation
    print("\n[MRI] Evaluating all methods …")
    results: Dict[str, Dict[str, float]] = {}

    print("  bicubic …")
    try:
        results["bicubic"] = _eval_bicubic_mri(MRI_ESRGAN_CONFIG, args.compute_lpips)
        print(f"    PSNR={results['bicubic']['psnr']:.2f}  SSIM={results['bicubic']['ssim']:.4f}")
    except Exception as e:
        print(f"    Error: {e}")

    for name, cfg_path in MRI_CONFIGS.items():
        print(f"  {name} …")
        ckpt = _best_checkpoint(cfg_path)
        if ckpt is None:
            print(f"    No checkpoint found — skipping (train first).")
            continue
        try:
            results[name] = _evaluate_checkpoint(cfg_path, str(ckpt), name, args.compute_lpips)
            r = results[name]
            print(f"    PSNR={r['psnr']:.2f}  SSIM={r['ssim']:.4f}"
                  + (f"  LPIPS={r.get('lpips', float('nan')):.4f}" if args.compute_lpips else ""))
        except Exception as e:
            print(f"    Error: {e}")

    # ESRGAN MRI
    print("  esrgan (existing run) …")
    esrgan_ckpt = None
    for p in [MRI_ESRGAN_BEST, MRI_ESRGAN_BEST_FALLBACK]:
        if Path(p).exists():
            esrgan_ckpt = p
            break
    if esrgan_ckpt:
        try:
            results["esrgan"] = _evaluate_checkpoint(
                MRI_ESRGAN_CONFIG, esrgan_ckpt, "esrgan", args.compute_lpips
            )
            r = results["esrgan"]
            print(f"    PSNR={r['psnr']:.2f}  SSIM={r['ssim']:.4f}"
                  + (f"  LPIPS={r.get('lpips', float('nan')):.4f}" if args.compute_lpips else ""))
        except Exception as e:
            print(f"    Error: {e}")
    else:
        print("    No ESRGAN MRI checkpoint found.")

    out_json = Path("runs/mri_evaluation.json")
    save_json(out_json, results)
    print(f"\nResults saved to {out_json}")

    try:
        _build_comparison_figure(results, Path("runs/mri_comparison.png"))
    except Exception as e:
        print(f"Warning: figure failed: {e}")

    print("\n[MRI] Done.")


if __name__ == "__main__":
    main()
