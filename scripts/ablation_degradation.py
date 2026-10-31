"""Experiment 2: Degradation ablation.

Trains ESRGAN under two identical configurations except for the degradation
pipeline and compares the results on the same validation set:

  Config A (bicubic-only):  configs/esrgan_ablation_bicubic.yaml
  Config B (mixed):         configs/esrgan_ablation_mixed.yaml

Both use the stable-GAN schedule for fair comparison.

Usage
-----
python scripts/ablation_degradation.py [--skip-training] [--compute-lpips]

  --skip-training   Use existing checkpoints; only regenerate results table
                    and qualitative figure. Useful after runs have finished.
  --compute-lpips   Include LPIPS in the results table (slower).

Outputs (written to runs/ablation_results/)
-------------------------------------------
  ablation_results.csv     — PSNR, SSIM, LPIPS per config
  qualitative_figure.png   — 4-panel side-by-side for 4 val images
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.core.device import DeviceManager
from src.dataset import SuperResolutionDataset
from src.metrics import batch_metrics
from src.models import build_model
from src.core.checkpoint import CheckpointManager
from src.utils import save_json

CONFIGS = {
    "bicubic_only": "configs/esrgan_ablation_bicubic.yaml",
    "mixed_degradation": "configs/esrgan_ablation_mixed.yaml",
}
OUTPUT_DIR = Path("runs/ablation_results")
NUM_QUALITATIVE = 4  # images used in the side-by-side figure


def _run_training(config_path: str) -> None:
    """Launch train.py as a subprocess for one config."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "train.py"),
        "--config", config_path,
    ]
    print(f"\n{'='*60}")
    print(f"Starting training: {config_path}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for config: {config_path}")


def _best_checkpoint(config_path: str) -> Optional[Path]:
    cfg = load_config(config_path)
    exp_dir = Path(cfg.save_dir) / cfg.experiment_name / "checkpoints"
    for name in ("best_generator.pth", "best.pth"):
        p = exp_dir / name
        if p.exists():
            return p
    return None


def _evaluate_config(config_path: str, compute_lpips: bool) -> Dict[str, float]:
    """Return mean PSNR, SSIM (and optionally LPIPS) on the val set."""
    ckpt_path = _best_checkpoint(config_path)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No best checkpoint found for {config_path}. "
            "Run training first or use --skip-training only after a completed run."
        )

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

    model = build_model(cfg.model_name, cfg.scale).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    all_metrics = []
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            all_metrics.append(batch_metrics(sr, hr, compute_lpips=compute_lpips))

    n = len(all_metrics)
    result: Dict[str, float] = {
        "psnr": float(sum(m["psnr"] for m in all_metrics) / n),
        "ssim": float(sum(m["ssim"] for m in all_metrics) / n),
        "num_images": float(n),
    }
    if compute_lpips and all("lpips" in m for m in all_metrics):
        result["lpips"] = float(sum(m["lpips"] for m in all_metrics) / n)
    DeviceManager.cleanup()
    return result


def _collect_sr_images(config_path: str, indices: List[int]) -> List[Image.Image]:
    """Run model inference on `indices` val images and return SR PIL images."""
    ckpt_path = _best_checkpoint(config_path)
    if ckpt_path is None:
        return []

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

    model = build_model(cfg.model_name, cfg.scale).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    images = []
    with torch.no_grad():
        for idx in indices:
            lr, _ = dataset[idx]
            sr = model(lr.unsqueeze(0).to(device))
            sr_np = sr[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy()
            images.append(Image.fromarray((sr_np * 255).astype("uint8")))

    DeviceManager.cleanup()
    return images


def _build_qualitative_figure(
    config_paths: Dict[str, str],
    val_dir: str,
    indices: List[int],
    output_path: Path,
) -> None:
    """Create an NUM_QUALITATIVE × (2+1)-column comparison figure."""
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    from src.dataset import list_images

    val_paths = list_images(val_dir)
    valid_indices = [i for i in indices if i < len(val_paths)]
    if not valid_indices:
        print("Warning: no valid validation indices for qualitative figure.")
        return

    n_cols = len(config_paths) + 1  # HR + one per config
    n_rows = len(valid_indices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = [axes]

    labels = ["HR Ground Truth"] + [name.replace("_", " ").title() for name in config_paths]
    sr_per_config = {}
    for name, cfg_path in config_paths.items():
        sr_per_config[name] = _collect_sr_images(cfg_path, valid_indices)

    for row, idx in enumerate(valid_indices):
        hr_path = val_paths[idx]
        hr_img = Image.open(hr_path).convert("RGB")
        # Use a central crop for display
        w, h = hr_img.size
        side = min(w, h, 256)
        left, top = (w - side) // 2, (h - side) // 2
        hr_crop = hr_img.crop((left, top, left + side, top + side))

        axes[row][0].imshow(hr_crop)
        axes[row][0].set_title(labels[0] if row == 0 else "")
        axes[row][0].axis("off")
        axes[row][0].set_ylabel(hr_path.stem[:20], fontsize=7)

        for col, (name, _) in enumerate(config_paths.items(), start=1):
            imgs = sr_per_config.get(name, [])
            if col - 1 < len(imgs):
                axes[row][col].imshow(imgs[col - 1])
            else:
                axes[row][col].text(0.5, 0.5, "N/A", ha="center", va="center")
            axes[row][col].set_title(labels[col] if row == 0 else "")
            axes[row][col].axis("off")

    plt.suptitle("Degradation Ablation: SR output comparison", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Qualitative figure saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Degradation ablation (Experiment 2)")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training; use existing checkpoints",
    )
    parser.add_argument(
        "--compute-lpips",
        action="store_true",
        help="Include LPIPS in results table",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_training:
        # Phase A: train bicubic config
        print("\n[Ablation] Training Config A: bicubic-only degradation")
        _run_training(CONFIGS["bicubic_only"])
        # Phase B: train mixed config
        print("\n[Ablation] Training Config B: mixed degradation")
        _run_training(CONFIGS["mixed_degradation"])
    else:
        print("[Ablation] Skipping training — using existing checkpoints.")

    # Evaluate
    print("\n[Ablation] Evaluating both configs...")
    results = {}
    for name, cfg_path in CONFIGS.items():
        try:
            r = _evaluate_config(cfg_path, compute_lpips=args.compute_lpips)
            results[name] = r
            print(f"  {name}: PSNR={r['psnr']:.2f}  SSIM={r['ssim']:.4f}"
                  + (f"  LPIPS={r.get('lpips', float('nan')):.4f}" if args.compute_lpips else ""))
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            results[name] = {}

    # Write CSV
    csv_path = OUTPUT_DIR / "ablation_results.csv"
    fieldnames = ["config", "psnr", "ssim", "lpips", "num_images"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, r in results.items():
            writer.writerow({
                "config": name,
                "psnr": f"{r.get('psnr', ''):.4f}" if r.get("psnr") else "",
                "ssim": f"{r.get('ssim', ''):.4f}" if r.get("ssim") else "",
                "lpips": f"{r.get('lpips', ''):.4f}" if r.get("lpips") else "",
                "num_images": int(r.get("num_images", 0)),
            })
    print(f"\nAblation results CSV: {csv_path}")

    # Save JSON too
    save_json(OUTPUT_DIR / "ablation_results.json", results)

    # Qualitative figure
    try:
        cfg_any = load_config(list(CONFIGS.values())[0])
        val_dir = cfg_any.dataset.val_hr_dir
        fig_path = OUTPUT_DIR / "qualitative_figure.png"
        _build_qualitative_figure(
            config_paths=CONFIGS,
            val_dir=val_dir,
            indices=list(range(NUM_QUALITATIVE)),
            output_path=fig_path,
        )
    except Exception as e:
        print(f"Warning: qualitative figure failed: {e}")

    print(f"\n[Ablation] Done. Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
