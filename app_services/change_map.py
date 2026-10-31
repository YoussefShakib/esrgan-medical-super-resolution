"""Super-Resolution Change Map (Enhancement Focus Region).

Highlights where the super-resolution method changed pixels most strongly
relative to a bicubic baseline. Returns a heatmap, an overlay, a bounding
box around the strongest changed region, and zoomed crops of both the
baseline and the final image at that region.

Explicitly non-diagnostic: this map indicates *where the model changed local
texture or edge detail most strongly*. It is not lesion detection, tumor
detection, or any clinical localisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ---------------------------------------------------------------------------
# Optional dependency: matplotlib for the colormap. Falls back to a manual
# blue→cyan→yellow→red ramp if matplotlib is missing, so the module never
# crashes the UI.
# ---------------------------------------------------------------------------

def _apply_colormap(gray: np.ndarray) -> np.ndarray:
    """Map a [0, 1] grayscale array to RGB uint8 via a jet-style colormap."""
    try:
        from matplotlib import cm  # type: ignore

        cmap = cm.get_cmap("jet")
        rgba = cmap(np.clip(gray, 0.0, 1.0))
        return (rgba[..., :3] * 255.0 + 0.5).astype(np.uint8)
    except Exception:
        # Manual fallback ramp — close enough visually to jet.
        g = np.clip(gray, 0.0, 1.0)
        r = np.clip(1.5 - np.abs(4 * g - 3), 0.0, 1.0)
        gg = np.clip(1.5 - np.abs(4 * g - 2), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4 * g - 1), 0.0, 1.0)
        rgb = np.stack([r, gg, b], axis=-1)
        return (rgb * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Connected-region bbox: prefer skimage; fall back to global bbox of all
# pixels above the threshold.
# ---------------------------------------------------------------------------

def _strongest_region_bbox(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    """Return (x, y, w, h) of the largest connected region of ``mask`` (bool)."""
    if mask.sum() == 0:
        return None
    try:
        from skimage.measure import label, regionprops  # type: ignore

        lbl = label(mask, connectivity=2)
        props = regionprops(lbl)
        if not props:
            return None
        biggest = max(props, key=lambda p: p.area)
        minr, minc, maxr, maxc = biggest.bbox  # rows, cols
        return (int(minc), int(minr), int(maxc - minc), int(maxr - minr))
    except Exception:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def _expand_bbox(
    box: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    margin_frac: float = 0.1,
    min_side: int = 32,
) -> Tuple[int, int, int, int]:
    """Slightly enlarge a bbox so the ROI crop has visual context."""
    x, y, w, h = box
    img_w, img_h = image_size
    margin = int(round(max(w, h) * margin_frac))
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img_w - x, w + 2 * margin)
    h = min(img_h - y, h + 2 * margin)
    if w < min_side:
        extra = (min_side - w) // 2
        x = max(0, x - extra)
        w = min(img_w - x, min_side)
    if h < min_side:
        extra = (min_side - h) // 2
        y = max(0, y - extra)
        h = min(img_h - y, min_side)
    return (x, y, w, h)


@dataclass
class ChangeMapResult:
    """Outputs of :func:`compute_change_map`."""

    diff_gray: np.ndarray  # float32 [0, 1], HxW
    heatmap_rgb: Image.Image  # PIL RGB
    overlay: Image.Image  # PIL RGB (final blended with heatmap)
    overlay_with_bbox: Image.Image  # overlay with a teal rectangle drawn around the bbox
    bbox: Tuple[int, int, int, int] | None  # (x, y, w, h) in final-image pixel coords
    roi_crop_baseline: Image.Image | None
    roi_crop_final: Image.Image | None
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_change_map(
    final_image: Image.Image,
    baseline_image: Image.Image,
    smooth_sigma: float = 4.0,
    bbox_top_percentile: float = 95.0,
    overlay_alpha: float = 0.45,
) -> ChangeMapResult:
    """Compute a super-resolution change map.

    Parameters
    ----------
    final_image
        The selected enhancement output (ESRGAN or hybrid).
    baseline_image
        Bicubic ×scale of the same LR. Resized to ``final_image.size`` if needed.
    smooth_sigma
        Gaussian smoothing radius in pixels for the difference map.
    bbox_top_percentile
        Pixels above this percentile of the smoothed diff are considered the
        "strongly changed" region; the largest connected component is the bbox.
    overlay_alpha
        Blend strength of the heatmap on top of the final image.
    """
    final_image = final_image.convert("RGB")
    baseline_image = baseline_image.convert("RGB")
    if baseline_image.size != final_image.size:
        baseline_image = baseline_image.resize(final_image.size, Image.LANCZOS)

    a = np.asarray(final_image, dtype=np.float32) / 255.0
    b = np.asarray(baseline_image, dtype=np.float32) / 255.0
    diff = np.abs(a - b).mean(axis=2)  # HxW float32 in [0, 1]

    # Smooth the difference so noise pixels don't dominate the bbox search.
    smoothed = _gaussian_smooth(diff, smooth_sigma)
    max_v = float(smoothed.max())
    norm = smoothed / max_v if max_v > 0 else smoothed.copy()

    threshold = float(np.percentile(norm, bbox_top_percentile))
    mask = norm >= threshold
    bbox_raw = _strongest_region_bbox(mask)

    heatmap_arr = _apply_colormap(norm)
    heatmap_pil = Image.fromarray(heatmap_arr, mode="RGB")

    overlay = Image.blend(final_image, heatmap_pil, overlay_alpha)

    bbox = _expand_bbox(bbox_raw, final_image.size) if bbox_raw is not None else None

    overlay_with_bbox = overlay.copy()
    if bbox is not None:
        draw = ImageDraw.Draw(overlay_with_bbox)
        x, y, w, h = bbox
        line_w = max(2, int(round(min(final_image.size) * 0.004)))
        draw.rectangle([x, y, x + w, y + h], outline=(79, 195, 184), width=line_w)

    roi_baseline = None
    roi_final = None
    if bbox is not None:
        x, y, w, h = bbox
        roi_baseline = baseline_image.crop((x, y, x + w, y + h))
        roi_final = final_image.crop((x, y, x + w, y + h))

    metadata: Dict[str, Any] = {
        "mean_diff": float(diff.mean()),
        "max_diff": float(diff.max()),
        "smoothed_max": max_v,
        "bbox_top_percentile": bbox_top_percentile,
        "smooth_sigma": smooth_sigma,
        "overlay_alpha": overlay_alpha,
        "bbox": list(bbox) if bbox is not None else None,
        "image_size": list(final_image.size),
    }

    return ChangeMapResult(
        diff_gray=norm.astype(np.float32),
        heatmap_rgb=heatmap_pil,
        overlay=overlay,
        overlay_with_bbox=overlay_with_bbox,
        bbox=bbox,
        roi_crop_baseline=roi_baseline,
        roi_crop_final=roi_final,
        metadata=metadata,
    )


def _gaussian_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur an HxW float array. Uses scipy if available, PIL otherwise."""
    if sigma <= 0:
        return arr
    try:
        from scipy.ndimage import gaussian_filter  # type: ignore

        return gaussian_filter(arr, sigma=float(sigma)).astype(np.float32)
    except Exception:
        # Fallback via PIL: scale to uint8, blur, scale back. Lower precision
        # but always available.
        max_v = float(arr.max())
        if max_v <= 0:
            return arr
        u8 = np.clip(arr / max_v * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(u8, mode="L").filter(ImageFilter.GaussianBlur(radius=float(sigma)))
        return (np.asarray(img, dtype=np.float32) / 255.0) * max_v
