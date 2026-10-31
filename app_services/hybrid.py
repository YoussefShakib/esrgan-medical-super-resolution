"""Hybrid Bicubic + ESRGAN super-resolution.

Combines a bicubic baseline with the learned ESRGAN detail. Bicubic gives a
mathematically smooth, predictable upscaling; ESRGAN supplies learned
high-frequency texture. The hybrid output keeps bicubic as the geometric base
and adds an alpha-weighted ESRGAN residual on top:

    detail   = esrgan_sr - bicubic_sr
    hybrid   = bicubic_sr + alpha * detail

`alpha = 0` reproduces bicubic; `alpha = 1` reproduces ESRGAN; values in
between trade ESRGAN sharpness against bicubic stability. The final image is
clamped to the valid pixel range [0, 1].

Important: this module never feeds a bicubic-upscaled image into the ESRGAN
model. ESRGAN already upscales by `scale`; both branches start from the same
LR input independently. Resizing only happens at the end if the two branch
outputs disagree by a few pixels (rare, but ESRGAN can drift if the LR size
is not perfectly divisible; the bicubic output defines the final size).
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def create_bicubic_output(image: Image.Image, scale: int = 4) -> Image.Image:
    """Bicubic x``scale`` upscale of ``image``. Output mode is forced to RGB."""
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    img = image.convert("RGB")
    w, h = img.size
    return img.resize((w * scale, h * scale), Image.BICUBIC)


def _to_float_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _to_pil_rgb(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def create_hybrid_bicubic_esrgan(
    input_image: Image.Image,
    esrgan_image: Image.Image,
    scale: int = 4,
    alpha: float = 0.8,
) -> Image.Image:
    """Blend bicubic and ESRGAN outputs by ``bicubic + alpha * (esrgan - bicubic)``.

    This helper supports two calling styles:

    1. Pass the original LR input as ``input_image`` plus ``scale``.
       The function will create the bicubic baseline internally.
    2. Pass a precomputed bicubic image as ``input_image``. If
       ``input_image.size == esrgan_image.size``, it is treated as the
       already-upscaled bicubic baseline and no second resize is applied.

    ``alpha`` is intentionally not clamped here. The final pixel values are
    always clamped to [0, 1] before returning.
    """
    input_image = input_image.convert("RGB")
    esrgan_image = esrgan_image.convert("RGB")

    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")

    if input_image.size == esrgan_image.size:
        bicubic_image = input_image
    else:
        bicubic_image = create_bicubic_output(input_image, scale=scale)

    if esrgan_image.size != bicubic_image.size:
        esrgan_image = esrgan_image.resize(bicubic_image.size, Image.LANCZOS)

    bic = _to_float_rgb(bicubic_image)
    esr = _to_float_rgb(esrgan_image)
    detail = esr - bic
    hybrid = bic + float(alpha) * detail
    return _to_pil_rgb(hybrid)


def compute_detail_residual(
    bicubic_image: Image.Image,
    esrgan_image: Image.Image,
) -> np.ndarray:
    """Return the absolute ESRGAN-vs-bicubic residual as a float32 [0, 1] HxW array.

    Useful for visualizing "what ESRGAN added" - the magnitude of learned
    texture above the bicubic baseline. Channels are mean-collapsed.
    """
    bicubic_image = bicubic_image.convert("RGB")
    esrgan_image = esrgan_image.convert("RGB")
    if esrgan_image.size != bicubic_image.size:
        esrgan_image = esrgan_image.resize(bicubic_image.size, Image.LANCZOS)

    bic = _to_float_rgb(bicubic_image)
    esr = _to_float_rgb(esrgan_image)
    diff = np.abs(esr - bic).mean(axis=2)
    max_v = float(diff.max())
    if max_v > 0:
        diff = diff / max_v
    return diff.astype(np.float32)
