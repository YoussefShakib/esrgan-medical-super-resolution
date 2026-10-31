"""Visual utilities for image comparison and presentation."""

from __future__ import annotations

from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


def default_lr_crop_box(lr_size: Tuple[int, int], crop: int = 256) -> Tuple[int, int, int, int]:
    """Return a centered (left, top, right, bottom) crop on the LR image."""
    w, h = lr_size
    side = min(w, h, crop)
    left = (w - side) // 2
    top = (h - side) // 2
    return (left, top, left + side, top + side)


def scale_crop_box(
    box: Tuple[int, int, int, int], scale: int
) -> Tuple[int, int, int, int]:
    """Scale an (l, t, r, b) crop box by an integer factor (e.g. LR→HR)."""
    return (box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale)


def make_triple_crop_panel(
    lr_image: Image.Image,
    bicubic_image: Image.Image,
    sr_image: Image.Image,
    lr_crop_box: Tuple[int, int, int, int],
    scale: int = 4,
    lr_label: str = "LR (input)",
    bicubic_label: str = "Bicubic ×4",
    sr_label: str = "ESRGAN ×4",
) -> Image.Image:
    """Three-panel side-by-side at native pixel ratio.

    - LR is displayed at `scale`× nearest-neighbour so pixels are visible at
      the same screen size as the SR crop (shows where detail is missing).
    - Bicubic and SR crops are shown at 1× (no extra resampling).
    All three output panels share the same pixel dimensions.
    """
    lr_crop = lr_image.crop(lr_crop_box)
    hr_box = scale_crop_box(lr_crop_box, scale)
    bic_crop = bicubic_image.crop(hr_box)
    sr_crop = sr_image.crop(hr_box)

    # Upscale LR with nearest so each LR pixel becomes a scale×scale block.
    lr_display = lr_crop.resize((lr_crop.width * scale, lr_crop.height * scale), Image.NEAREST)

    left = _add_label_band(lr_display, lr_label)
    mid = _add_label_band(bic_crop, bicubic_label)
    right = _add_label_band(sr_crop, sr_label)

    panel_h = max(left.height, mid.height, right.height)
    gap = 8
    panel = Image.new("RGB", (left.width + mid.width + right.width + 2 * gap, panel_h), (14, 17, 23))
    panel.paste(left, (0, 0))
    panel.paste(mid, (left.width + gap, 0))
    panel.paste(right, (left.width + mid.width + 2 * gap, 0))
    return panel


def _get_default_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Get a default font, falling back to PIL default if not available.
    
    Args:
        size: Font size
        
    Returns:
        Font object
    """
    try:
        # Try to use a common system font
        return ImageFont.truetype("arial.ttf", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except (OSError, IOError):
            # Fall back to default font
            return ImageFont.load_default()


def resize_for_display(image: Image.Image, max_side: int = 1024) -> Image.Image:
    """
    Resize image for display while maintaining aspect ratio.
    
    Args:
        image: PIL Image
        max_side: Maximum dimension size
        
    Returns:
        Resized PIL Image
        
    Raises:
        ValueError: If image is invalid
    """
    try:
        image = image.convert("RGB")
        width, height = image.size
        longest_side = max(width, height)

        if longest_side <= max_side:
            return image

        scale = max_side / float(longest_side)
        new_size = (int(width * scale), int(height * scale))
        return image.resize(new_size, Image.LANCZOS)
    except Exception as e:
        raise ValueError(f"Failed to resize image: {str(e)}") from e


def upscale_for_visual_comparison(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """
    Upscale image to target size for comparison.
    
    Args:
        image: PIL Image
        target_size: Target (width, height)
        
    Returns:
        Upscaled PIL Image
        
    Raises:
        ValueError: If upscaling fails
    """
    try:
        return image.convert("RGB").resize(target_size, Image.BICUBIC)
    except Exception as e:
        raise ValueError(f"Failed to upscale image: {str(e)}") from e


def _add_label_band(image: Image.Image, label: str, band_height: int = 36) -> Image.Image:
    """
    Add a label band at the top of an image.
    
    Args:
        image: PIL Image
        label: Label text
        band_height: Height of label band
        
    Returns:
        Image with label band
        
    Raises:
        RuntimeError: If labeling fails
    """
    try:
        image = image.convert("RGB")
        canvas = Image.new("RGB", (image.width, image.height + band_height), (14, 17, 23))
        canvas.paste(image, (0, band_height))

        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, canvas.width, band_height], fill=(22, 26, 35))
        font = _get_default_font(14)
        draw.text((12, 10), label, fill=(228, 231, 235), font=font)

        return canvas
    except Exception as e:
        raise RuntimeError(f"Failed to add label band: {str(e)}") from e


def make_side_by_side_panel(
    input_image: Image.Image,
    output_image: Image.Image,
    input_label: str = "Input",
    output_label: str = "ESRGAN Output",
) -> Image.Image:
    """
    Create side-by-side comparison panel.
    
    Args:
        input_image: Input/reference image
        output_image: Output/enhanced image
        input_label: Label for input image
        output_label: Label for output image
        
    Returns:
        Comparison panel image
        
    Raises:
        ValueError: If images are invalid
        RuntimeError: If panel creation fails
    """
    try:
        input_resized = upscale_for_visual_comparison(input_image, output_image.size)
        left = _add_label_band(input_resized, input_label)
        right = _add_label_band(output_image, output_label)

        panel = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), "white")
        panel.paste(left, (0, 0))
        panel.paste(right, (left.width, 0))
        return panel
    except Exception as e:
        raise RuntimeError(f"Failed to create comparison panel: {str(e)}") from e


def make_zoom_panel(
    input_image: Image.Image,
    output_image: Image.Image,
    crop_box: Optional[Tuple[int, int, int, int]] = None,
    input_label: str = "Input Crop",
    output_label: str = "ESRGAN Crop",
) -> Image.Image:
    """
    Create zoomed comparison panel.
    
    Args:
        input_image: Input/reference image
        output_image: Output/enhanced image
        crop_box: Optional crop box (left, top, right, bottom)
        input_label: Label for input crop
        output_label: Label for output crop
        
    Returns:
        Zoom comparison panel image
        
    Raises:
        ValueError: If images or crop box invalid
        RuntimeError: If panel creation fails
    """
    try:
        if crop_box is None:
            w, h = input_image.size
            crop_size = min(w, h, 256)
            left = max(0, (w - crop_size) // 2)
            top = max(0, (h - crop_size) // 2)
            crop_box = (left, top, left + crop_size, top + crop_size)

        # Validate crop box
        if len(crop_box) != 4 or not all(isinstance(x, int) for x in crop_box):
            raise ValueError(f"Invalid crop box: {crop_box}")
        
        input_crop = input_image.crop(crop_box)

        scale_x = output_image.width / input_image.width
        scale_y = output_image.height / input_image.height
        ox1 = int(crop_box[0] * scale_x)
        oy1 = int(crop_box[1] * scale_y)
        ox2 = int(crop_box[2] * scale_x)
        oy2 = int(crop_box[3] * scale_y)
        output_crop = output_image.crop((ox1, oy1, ox2, oy2))

        output_crop = output_crop.resize((input_crop.width * 2, input_crop.height * 2), Image.LANCZOS)
        input_crop = input_crop.resize(output_crop.size, Image.BICUBIC)

        return make_side_by_side_panel(
            input_crop,
            output_crop,
            input_label=input_label,
            output_label=output_label,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create zoom panel: {str(e)}") from e