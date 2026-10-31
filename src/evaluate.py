import argparse
import sys
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained SR model")
    parser.add_argument("--config", required=True, help="YAML config used for training")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pth)")
    parser.add_argument(
        "--compute-lpips",
        action="store_true",
        default=False,
        help="Include LPIPS in results (slower; requires lpips package)",
    )
    parser.add_argument("--output", default=None, help="Override output JSON path")
    args = parser.parse_args()

    config = load_config(args.config)
    device = DeviceManager.setup(config.model_dump())

    srcnn_mode = config.model_name.lower() == "srcnn"
    dataset = SuperResolutionDataset(
        hr_dir=config.dataset.val_hr_dir,
        scale=config.scale,
        patch_size=config.dataset.patch_size,
        is_train=False,
        srcnn_mode=srcnn_mode,
        degradation_config={
            **config.dataset.degradation,
            "eval_crop_size": config.dataset.eval_crop_size,
        },
        image_mode=config.dataset.image_mode,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(config.model_name, config.scale).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    all_metrics = []
    with torch.no_grad():
        for lr, hr in loader:
            lr = lr.to(device)
            hr = hr.to(device)
            sr = model(lr)
            m = batch_metrics(sr, hr, compute_lpips=args.compute_lpips)
            all_metrics.append(m)

    n = len(all_metrics)
    results = {
        "checkpoint": args.checkpoint,
        "num_images": n,
        "mean_psnr": float(sum(m["psnr"] for m in all_metrics) / n),
        "mean_ssim": float(sum(m["ssim"] for m in all_metrics) / n),
    }
    if args.compute_lpips and all("lpips" in m for m in all_metrics):
        results["mean_lpips"] = float(sum(m["lpips"] for m in all_metrics) / n)

    out_path = Path(args.output) if args.output else Path(args.checkpoint).with_name("evaluation_results.json")
    save_json(out_path, results)
    print(results)
    DeviceManager.cleanup()


if __name__ == "__main__":
    main()
