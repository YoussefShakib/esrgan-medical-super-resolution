"""One-off Phase-1 sanity check.

Runs the new UI metric flow on a handful of BACH val images:
    HR (upload) -> bicubic x1/4 -> LR
    LR -> ESRGAN -> SR
    LR -> bicubic x4 -> Bicubic baseline
    PSNR/SSIM (RGB, crop=4) of SR vs HR and Bicubic vs HR

Run:
    python tests/check_ui_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from PIL import Image

from app_services import load_sr_service
from app_services.metrics_util import (
    align_hr_for_scale,
    bicubic_downsample,
    bicubic_upsample,
    compute_psnr,
    compute_ssim,
)


CONFIG = REPO_ROOT / "configs" / "esrgan_bach_medical.yaml"
CHECKPOINT = REPO_ROOT / "runs" / "esrgan_bach_x4" / "checkpoints" / "best_generator.pth"
VAL_DIR = REPO_ROOT / "datasets" / "val_HR"
SCALE = 4
CROP = 4
EVAL_CROP = 512
N_IMAGES = 5


def center_crop(image: Image.Image, size: int) -> Image.Image:
    w, h = image.size
    side = min(w, h, size)
    left = (w - side) // 2
    top = (h - side) // 2
    return image.crop((left, top, left + side, top + side))


def main() -> None:
    service = load_sr_service(str(CONFIG), str(CHECKPOINT))

    val_paths = sorted(VAL_DIR.glob("*.tif"))[:N_IMAGES]
    print(f"Evaluating {len(val_paths)} images from {VAL_DIR}\n")

    header = f"{'image':<28} {'bicubic PSNR':>13} {'ESRGAN PSNR':>13} {'bicubic SSIM':>13} {'ESRGAN SSIM':>13}"
    print(header)
    print("-" * len(header))

    psnr_bic, psnr_sr, ssim_bic, ssim_sr = [], [], [], []
    for p in val_paths:
        hr = Image.open(p).convert("RGB")
        hr = center_crop(hr, EVAL_CROP)
        hr = align_hr_for_scale(hr, SCALE)

        lr = bicubic_downsample(hr, SCALE)
        bicubic = bicubic_upsample(lr, SCALE)

        sr = service.enhance_pil_image(lr, use_tiling=True, tile_size=128, tile_overlap=24)

        pb = compute_psnr(bicubic, hr, crop=CROP)
        ps = compute_psnr(sr, hr, crop=CROP)
        sb = compute_ssim(bicubic, hr, crop=CROP)
        ss = compute_ssim(sr, hr, crop=CROP)

        psnr_bic.append(pb); psnr_sr.append(ps); ssim_bic.append(sb); ssim_sr.append(ss)
        print(f"{p.name:<28} {pb:>13.3f} {ps:>13.3f} {sb:>13.4f} {ss:>13.4f}")

    print("-" * len(header))
    n = len(val_paths)
    print(
        f"{'MEAN':<28} "
        f"{sum(psnr_bic)/n:>13.3f} "
        f"{sum(psnr_sr)/n:>13.3f} "
        f"{sum(ssim_bic)/n:>13.4f} "
        f"{sum(ssim_sr)/n:>13.4f}"
    )
    print(
        f"\nConvention: RGB, border crop = {CROP} px, float32 [0,1], data_range=1.0"
        f"\nEval crop (HR): center {EVAL_CROP}x{EVAL_CROP}, aligned to scale {SCALE}"
    )


if __name__ == "__main__":
    main()
