"""Compute the bicubic baseline PSNR/SSIM/LPIPS on the val set.

This uses the SAME evaluation protocol as src/evaluate.py and
src/train.py: bicubic downsampling at evaluation time, eval_crop_size=512,
image_mode=RGB — so the number is directly comparable to the trained-model
evaluation_results.json files.

Usage
-----
    python scripts/eval_bicubic_baseline.py [--config CONFIG] [--compute-lpips]

Writes results to runs/bicubic_baseline/evaluation_results.json.
"""

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.core.device import DeviceManager
from src.dataset import SuperResolutionDataset
from src.metrics import batch_metrics
from src.utils import save_json

DEFAULT_CONFIG = "configs/esrgan_bach_medical.yaml"


def bicubic_sr(lr: torch.Tensor, scale: int) -> torch.Tensor:
    """Bicubic upsampling to simulate the non-learned baseline."""
    h, w = lr.shape[-2], lr.shape[-1]
    return TF.resize(
        lr,
        size=[h * scale, w * scale],
        interpolation=TF.InterpolationMode.BICUBIC,
        antialias=True,
    ).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(description="Bicubic baseline evaluation")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--compute-lpips",
        action="store_true",
        default=False,
        help="Include LPIPS (slower)",
    )
    parser.add_argument("--output-dir", default="runs/bicubic_baseline")
    args = parser.parse_args()

    config = load_config(args.config)
    device = DeviceManager.setup(config.model_dump())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = SuperResolutionDataset(
        hr_dir=config.dataset.val_hr_dir,
        scale=config.scale,
        patch_size=config.dataset.patch_size,
        is_train=False,
        srcnn_mode=False,
        degradation_config={
            **config.dataset.degradation,
            "eval_crop_size": config.dataset.eval_crop_size,
        },
        image_mode=config.dataset.image_mode,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Evaluating bicubic ×{config.scale} on {len(dataset)} val images …")

    all_metrics = []
    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        bic = bicubic_sr(lr, config.scale)
        m = batch_metrics(bic, hr, compute_lpips=args.compute_lpips)
        all_metrics.append(m)

    n = len(all_metrics)
    results = {
        "method": "bicubic",
        "scale": config.scale,
        "val_hr_dir": config.dataset.val_hr_dir,
        "eval_crop_size": config.dataset.eval_crop_size,
        "num_images": n,
        "mean_psnr": float(sum(m["psnr"] for m in all_metrics) / n),
        "mean_ssim": float(sum(m["ssim"] for m in all_metrics) / n),
    }
    if args.compute_lpips and all("lpips" in m for m in all_metrics):
        results["mean_lpips"] = float(sum(m["lpips"] for m in all_metrics) / n)

    out_path = out_dir / "evaluation_results.json"
    save_json(out_path, results)

    print(f"\n{'='*50}")
    print(f"Bicubic ×{config.scale} — {n} images")
    print(f"  PSNR : {results['mean_psnr']:.4f} dB")
    print(f"  SSIM : {results['mean_ssim']:.4f}")
    if "mean_lpips" in results:
        print(f"  LPIPS: {results['mean_lpips']:.4f}")
    print(f"  Written to: {out_path}")
    print(f"{'='*50}")

    DeviceManager.cleanup()


if __name__ == "__main__":
    main()
