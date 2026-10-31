"""Hybrid Bicubic + ESRGAN experiment.

For every HR image in ``--val-dir`` (or for a single ``--val-dir`` image
path), this script:

    1. Aligns HR to a multiple of ``scale``.
    2. Generates LR via deterministic bicubic ÷scale.
    3. Computes:
        - bicubic_sr  = bicubic ×scale of LR
        - esrgan_sr   = ESRGAN ×scale of LR
        - hybrid_sr   = bicubic_sr + alpha * (esrgan_sr - bicubic_sr)
    4. Measures PSNR / SSIM (RGB, crop=4) of each method against HR.
    5. Saves LR, bicubic, ESRGAN, hybrid, residual heatmap, and a labelled
       comparison panel for every image.
    6. Writes ``metrics.json`` with per-image and aggregate scores.

ESRGAN already upscales by ``scale`` — we never feed bicubic-upscaled images
into the ESRGAN model. Both branches start from the same LR.

Examples
--------

    python src/experiment_hybrid_bicubic_esrgan.py \
        --config configs/esrgan_bach_medical.yaml \
        --checkpoint runs/esrgan_bach_x4/checkpoints/best_generator.pth \
        --val-dir datasets/val_HR \
        --output-dir thesis_outputs/bach_hybrid_bicubic_esrgan \
        --alpha 0.8

    python src/experiment_hybrid_bicubic_esrgan.py \
        --config configs/esrgan_brain_mri.yaml \
        --checkpoint runs/esrgan_brain_mri_x4/checkpoints/best_generator.pth \
        --val-dir datasets/BRAIN_MRI/val_HR \
        --output-dir thesis_outputs/brain_mri_hybrid_bicubic_esrgan \
        --alpha 0.8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Make sibling app_services package importable when running this file directly
# (mirrors what app_services/__init__.py does for src/* imports).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from PIL import Image

from app_services import (
    align_hr_for_scale,
    bicubic_downsample,
    compute_psnr,
    compute_ssim,
    compute_detail_residual,
    create_bicubic_output,
    create_hybrid_bicubic_esrgan,
    load_sr_service,
    make_triple_crop_panel,
)
from app_services.change_map import _apply_colormap  # internal but reusable here


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_inputs(val_dir: Path | None = None, input_image: Path | None = None) -> List[Path]:
    if input_image is not None:
        if not input_image.exists() or not input_image.is_file():
            raise FileNotFoundError(f"--input-image not found: {input_image}")
        return [input_image]
    if val_dir is None:
        raise ValueError("Either --val-dir or --input-image must be provided.")
    if val_dir.is_file():
        return [val_dir]
    if not val_dir.is_dir():
        raise FileNotFoundError(f"--val-dir not found: {val_dir}")
    return sorted(p for p in val_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS)


def _save_residual_heatmap(residual: np.ndarray, path: Path) -> None:
    rgb = _apply_colormap(residual)
    Image.fromarray(rgb, mode="RGB").save(path)


def _build_panel(
    hr: Image.Image,
    lr: Image.Image,
    bicubic_sr: Image.Image,
    esrgan_sr: Image.Image,
    hybrid_sr: Image.Image,
    scale: int,
    alpha: float,
) -> Image.Image:
    """Two stacked rows: top row LR / bicubic / esrgan, bottom row hybrid / hr."""
    from app_services.visuals import _add_label_band  # type: ignore

    top = make_triple_crop_panel(
        lr_image=lr,
        bicubic_image=bicubic_sr,
        sr_image=esrgan_sr,
        lr_crop_box=(0, 0, lr.width, lr.height),
        scale=scale,
        lr_label=f"LR (×{scale} nearest)",
        bicubic_label=f"Bicubic ×{scale}",
        sr_label="ESRGAN",
    )
    hybrid_panel = _add_label_band(hybrid_sr, f"Hybrid α={alpha:.2f}")
    hr_panel = _add_label_band(hr, "HR (ground truth)")
    bottom_h = max(hybrid_panel.height, hr_panel.height)
    gap = 8
    bottom = Image.new(
        "RGB", (hybrid_panel.width + hr_panel.width + gap, bottom_h), (14, 17, 23)
    )
    bottom.paste(hybrid_panel, (0, 0))
    bottom.paste(hr_panel, (hybrid_panel.width + gap, 0))
    panel = Image.new(
        "RGB",
        (max(top.width, bottom.width), top.height + bottom.height + gap),
        (14, 17, 23),
    )
    panel.paste(top, (0, 0))
    panel.paste(bottom, (0, top.height + gap))
    return panel


def run(
    config_path: Path,
    checkpoint_path: Path,
    val_dir: Path | None,
    input_image: Path | None,
    output_dir: Path,
    alpha: float,
    scale: int,
    max_images: int | None,
    use_tiling: bool,
    tile_size: int,
    tile_overlap: int,
) -> None:
    inputs = _list_inputs(val_dir=val_dir, input_image=input_image)
    if max_images is not None:
        inputs = inputs[:max_images]
    if not inputs:
        source_label = input_image if input_image is not None else val_dir
        raise RuntimeError(f"No images found in {source_label} (extensions: {sorted(VALID_EXTS)}).")

    print(f"Loading SR service from {checkpoint_path}…")
    service = load_sr_service(str(config_path), str(checkpoint_path))
    cfg_scale = int(service.config.get("scale", scale))
    if cfg_scale != scale:
        print(
            f"Warning: --scale={scale} differs from config scale={cfg_scale}; "
            f"using config scale.",
        )
        scale = cfg_scale

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_image: List[Dict[str, Any]] = []

    t0 = time.time()
    for idx, hr_path in enumerate(inputs, 1):
        stem = hr_path.stem
        case_dir = output_dir / stem
        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{idx}/{len(inputs)}] {hr_path.name}")

        try:
            hr = align_hr_for_scale(Image.open(hr_path).convert("RGB"), scale)
            lr = bicubic_downsample(hr, scale)
            bicubic_sr = create_bicubic_output(lr, scale=scale)
            esrgan_sr = service.enhance_pil_image(
                lr,
                use_tiling=use_tiling,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
            )
            hybrid_sr = create_hybrid_bicubic_esrgan(lr, esrgan_sr, scale=scale, alpha=alpha)

            # Metrics — PSNR/SSIM helpers align internally if sizes drift.
            metrics = {
                "bicubic": {
                    "psnr": compute_psnr(bicubic_sr, hr, crop=4),
                    "ssim": compute_ssim(bicubic_sr, hr, crop=4),
                },
                "esrgan": {
                    "psnr": compute_psnr(esrgan_sr, hr, crop=4),
                    "ssim": compute_ssim(esrgan_sr, hr, crop=4),
                },
                "hybrid": {
                    "psnr": compute_psnr(hybrid_sr, hr, crop=4),
                    "ssim": compute_ssim(hybrid_sr, hr, crop=4),
                },
            }

            # Save artifacts.
            lr.save(case_dir / "lr_input.png")
            bicubic_sr.save(case_dir / "bicubic_output.png")
            esrgan_sr.save(case_dir / "esrgan_output.png")
            hybrid_sr.save(case_dir / "hybrid_output.png")
            hr.save(case_dir / "hr_reference.png")
            residual = compute_detail_residual(bicubic_sr, esrgan_sr)
            _save_residual_heatmap(residual, case_dir / "detail_residual_map.png")
            panel = _build_panel(hr, lr, bicubic_sr, esrgan_sr, hybrid_sr, scale, alpha)
            panel.save(case_dir / "comparison_panel.png")
            # Backward-compatible legacy names.
            panel.save(case_dir / "panel.png")
            _save_residual_heatmap(residual, case_dir / "residual.png")
            (case_dir / "metrics.json").write_text(
                json.dumps({"image": hr_path.name, "alpha": alpha, "metrics": metrics}, indent=2),
                encoding="utf-8",
            )

            per_image.append(
                {
                    "image": hr_path.name,
                    "size": [hr.width, hr.height],
                    "metrics": metrics,
                }
            )
            print(
                f"   bicubic PSNR {metrics['bicubic']['psnr']:6.3f}  "
                f"esrgan PSNR {metrics['esrgan']['psnr']:6.3f}  "
                f"hybrid PSNR {metrics['hybrid']['psnr']:6.3f}"
            )
        except Exception as e:
            print(f"   ! failed: {e}")
            per_image.append({"image": hr_path.name, "error": str(e)})

    # Aggregate.
    successful = [r for r in per_image if "metrics" in r]
    aggregate: Dict[str, Dict[str, float]] = {}
    if successful:
        for method in ("bicubic", "esrgan", "hybrid"):
            psnrs = [r["metrics"][method]["psnr"] for r in successful]
            ssims = [r["metrics"][method]["ssim"] for r in successful]
            aggregate[method] = {
                "psnr_mean": float(np.mean(psnrs)),
                "psnr_std": float(np.std(psnrs)),
                "ssim_mean": float(np.mean(ssims)),
                "ssim_std": float(np.std(ssims)),
            }

    summary = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "val_dir": str(val_dir) if val_dir is not None else None,
        "input_image": str(input_image) if input_image is not None else None,
        "alpha": alpha,
        "scale": scale,
        "num_images": len(per_image),
        "num_successful": len(successful),
        "aggregate": aggregate,
        "per_image": per_image,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"Wrote {len(successful)}/{len(per_image)} cases to {output_dir}")
    if aggregate:
        print("Aggregate (mean ± std):")
        for method, stats in aggregate.items():
            print(
                f"  {method:>7}  PSNR {stats['psnr_mean']:6.3f} ± {stats['psnr_std']:.3f}   "
                f"SSIM {stats['ssim_mean']:.4f} ± {stats['ssim_std']:.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid bicubic + ESRGAN super-resolution experiment.")
    parser.add_argument("--config", required=True, help="Path to model config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Path to ESRGAN generator checkpoint.")
    parser.add_argument("--val-dir", default=None, help="Folder of HR validation images, or a single HR image path.")
    parser.add_argument("--input-image", default=None, help="Optional single HR image path.")
    parser.add_argument("--output-dir", required=True, help="Where to write per-image folders + metrics.json.")
    parser.add_argument("--alpha", type=float, default=0.8, help="ESRGAN detail strength (0=bicubic, 1=ESRGAN).")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    parser.add_argument("--no-tiling", action="store_true", help="Disable tiled inference.")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-overlap", type=int, default=24)
    args = parser.parse_args()

    run(
        config_path=Path(args.config),
        checkpoint_path=Path(args.checkpoint),
        val_dir=Path(args.val_dir) if args.val_dir else None,
        input_image=Path(args.input_image) if args.input_image else None,
        output_dir=Path(args.output_dir),
        alpha=args.alpha,
        scale=args.scale,
        max_images=args.max_images,
        use_tiling=not args.no_tiling,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
    )


if __name__ == "__main__":
    main()
