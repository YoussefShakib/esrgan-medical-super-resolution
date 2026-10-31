"""Reference-based image-quality metrics for the live UI.

Convention (matches training-time `src/metrics.py` with a light boundary crop
added for fairness):
    - Color space: full RGB (3 channels)
    - Range: float32 in [0, 1]
    - Border crop: `scale` pixels (default 4) on every side before measurement
    - SSIM `data_range=1.0`, channel_axis=2
    - LPIPS: AlexNet backbone, inputs scaled to [-1, 1], no border crop

The boundary crop prevents tile / padding artefacts at image edges from
dominating the score; this is standard in the SR literature (EDSR, ESRGAN).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_lpips_net: Optional[object] = None


def _get_lpips_net():
    global _lpips_net
    if _lpips_net is None:
        try:
            import lpips as _lib
            _lpips_net = _lib.LPIPS(net="alex", verbose=False)
        except ImportError:
            return None
    return _lpips_net


def _pil_to_float_rgb(image: Image.Image) -> np.ndarray:
    """PIL → float32 RGB array in [0, 1]."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def _align_to_reference(image: Image.Image, reference: Image.Image) -> Image.Image:
    """Resize `image` to match `reference` size if they differ by a small amount."""
    if image.size == reference.size:
        return image
    return image.resize(reference.size, Image.BICUBIC)


def _crop_border(arr: np.ndarray, crop: int) -> np.ndarray:
    """Crop `crop` pixels from every spatial border."""
    if crop <= 0:
        return arr
    h, w = arr.shape[:2]
    if h <= 2 * crop or w <= 2 * crop:
        return arr
    return arr[crop : h - crop, crop : w - crop, ...]


def compute_psnr(sr: Image.Image, hr: Image.Image, crop: int = 4) -> float:
    """PSNR (RGB, crop=`crop`) in dB. Sizes must match after alignment."""
    sr = _align_to_reference(sr, hr)
    sr_np = _crop_border(_pil_to_float_rgb(sr), crop)
    hr_np = _crop_border(_pil_to_float_rgb(hr), crop)
    return float(peak_signal_noise_ratio(hr_np, sr_np, data_range=1.0))


def compute_ssim(sr: Image.Image, hr: Image.Image, crop: int = 4) -> float:
    """SSIM (RGB, crop=`crop`). Sizes must match after alignment."""
    sr = _align_to_reference(sr, hr)
    sr_np = _crop_border(_pil_to_float_rgb(sr), crop)
    hr_np = _crop_border(_pil_to_float_rgb(hr), crop)
    return float(
        structural_similarity(hr_np, sr_np, channel_axis=2, data_range=1.0)
    )


def bicubic_downsample(hr: Image.Image, scale: int = 4) -> Image.Image:
    """Create a matched-pipeline LR image (bicubic ÷scale, crop to multiple of scale)."""
    w, h = hr.size
    w = w - (w % scale)
    h = h - (h % scale)
    if w == 0 or h == 0:
        raise ValueError(f"HR too small for scale {scale}: {hr.size}")
    hr_cropped = hr.crop((0, 0, w, h))
    lr = hr_cropped.resize((w // scale, h // scale), Image.BICUBIC)
    return lr


def bicubic_upsample(lr: Image.Image, scale: int = 4) -> Image.Image:
    """Bicubic ×scale upsample, used as the non-learned baseline."""
    w, h = lr.size
    return lr.resize((w * scale, h * scale), Image.BICUBIC)


def align_hr_for_scale(hr: Image.Image, scale: int = 4) -> Image.Image:
    """Crop HR so both dimensions are multiples of `scale`. Identity if already aligned."""
    w, h = hr.size
    new_w = w - (w % scale)
    new_h = h - (h % scale)
    if new_w == w and new_h == h:
        return hr
    return hr.crop((0, 0, new_w, new_h))


def compute_lpips(sr: Image.Image, hr: Image.Image) -> Optional[float]:
    """LPIPS perceptual distance (AlexNet, lower = more similar to HR).

    Inputs: PIL RGB images, same size. No border crop (matches thesis convention).
    Returns None if lpips package is not installed.
    """
    net = _get_lpips_net()
    if net is None:
        return None
    sr = _align_to_reference(sr, hr)

    def _to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return t * 2.0 - 1.0  # lpips expects [-1, 1]

    with torch.no_grad():
        score = net(_to_tensor(sr), _to_tensor(hr))
    return float(score.mean().item())


def reference_metrics_table(
    hr: Image.Image,
    sr: Image.Image,
    bicubic: Image.Image,
    crop: int = 4,
    include_lpips: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Compute the `Bicubic | ESRGAN` row pair the UI displays.

    Returns a dict keyed by method, each value containing psnr, ssim,
    and optionally lpips (when include_lpips=True and the package is available).
    """
    table: Dict[str, Dict[str, float]] = {
        "bicubic": {
            "psnr": compute_psnr(bicubic, hr, crop=crop),
            "ssim": compute_ssim(bicubic, hr, crop=crop),
        },
        "esrgan": {
            "psnr": compute_psnr(sr, hr, crop=crop),
            "ssim": compute_ssim(sr, hr, crop=crop),
        },
    }
    if include_lpips:
        bic_lpips = compute_lpips(bicubic, hr)
        sr_lpips = compute_lpips(sr, hr)
        if bic_lpips is not None:
            table["bicubic"]["lpips"] = bic_lpips
        if sr_lpips is not None:
            table["esrgan"]["lpips"] = sr_lpips
    return table
