"""App services for medical image super-resolution."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from .sr_service import SuperResolutionService, load_sr_service
from .image_analysis import analyze_image, compare_analysis, summarize_analysis, ImageAnalysis
from .findings import generate_findings_report, findings_to_markdown
from .visuals import (
    resize_for_display,
    upscale_for_visual_comparison,
    make_side_by_side_panel,
    make_zoom_panel,
    make_triple_crop_panel,
    default_lr_crop_box,
    scale_crop_box,
)
from .metrics_util import (
    compute_psnr,
    compute_ssim,
    bicubic_downsample,
    bicubic_upsample,
    align_hr_for_scale,
    reference_metrics_table,
)
from .hybrid import (
    create_bicubic_output,
    create_hybrid_bicubic_esrgan,
    compute_detail_residual,
)
from .change_map import compute_change_map, ChangeMapResult
from .classifier_service import ClassifierService, PredictionResult, load_classifier_service

__all__ = [
    "PROJECT_ROOT",
    "SRC_DIR",
    "SuperResolutionService",
    "load_sr_service",
    "analyze_image",
    "compare_analysis",
    "summarize_analysis",
    "ImageAnalysis",
    "generate_findings_report",
    "findings_to_markdown",
    "resize_for_display",
    "upscale_for_visual_comparison",
    "make_side_by_side_panel",
    "make_zoom_panel",
    "make_triple_crop_panel",
    "default_lr_crop_box",
    "scale_crop_box",
    "compute_psnr",
    "compute_ssim",
    "bicubic_downsample",
    "bicubic_upsample",
    "align_hr_for_scale",
    "reference_metrics_table",
    "create_bicubic_output",
    "create_hybrid_bicubic_esrgan",
    "compute_detail_residual",
    "compute_change_map",
    "ChangeMapResult",
    "ClassifierService",
    "PredictionResult",
    "load_classifier_service",
]
