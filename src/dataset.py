from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageFilter
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


PIL_RESAMPLING = {
    "bicubic": Image.BICUBIC,
    "bilinear": Image.BILINEAR,
    "nearest": Image.NEAREST,
    "lanczos": Image.LANCZOS,
}

TORCH_INTERPOLATION = {
    "bicubic": InterpolationMode.BICUBIC,
    "bilinear": InterpolationMode.BILINEAR,
    "nearest": InterpolationMode.NEAREST,
    "lanczos": InterpolationMode.BICUBIC,
}


def list_images(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in VALID_EXTS])


def paired_random_crop(image: Image.Image, patch_size: int) -> Image.Image:
    width, height = image.size
    patch_size = min(patch_size, width, height)
    left = random.randint(0, max(0, width - patch_size))
    top = random.randint(0, max(0, height - patch_size))
    return image.crop((left, top, left + patch_size, top + patch_size))


def augment_pair(image: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        image = TF.hflip(image)
    if random.random() < 0.5:
        image = TF.vflip(image)
    rotations = random.choice([0, 90, 180, 270])
    if rotations:
        image = image.rotate(rotations)
    return image


def _apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """Apply JPEG compression degradation. Fixes memory leak by closing buffer."""
    buffer = BytesIO()
    try:
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        # Load into new image and close buffer to prevent memory leak
        compressed = Image.open(buffer)
        # Convert to ensure image is loaded before buffer closes
        result = compressed.convert(image.mode).copy()
        buffer.close()
        return result
    except Exception as e:
        buffer.close()
        raise RuntimeError(f"JPEG compression failed: {str(e)}") from e


def _degradation_method(config: Dict[str, Any], is_train: bool) -> str:
    method = str(config.get("method", "bicubic")).lower()
    if method != "mixed" or not is_train:
        return method

    choices = config.get("mixed_methods", ["bicubic", "bilinear", "nearest"])
    valid_choices = [str(choice).lower() for choice in choices if str(choice).lower() in PIL_RESAMPLING]
    return random.choice(valid_choices or ["bicubic"])


def make_lr_from_hr(
    hr_image: Image.Image,
    scale: int,
    degradation_config: Dict[str, Any] | None = None,
    is_train: bool = True
) -> Image.Image:
    """
    Degrade HR image to LR version with optional degradations.
    
    Args:
        hr_image: High-resolution PIL image
        scale: Downsampling scale factor
        degradation_config: Dict with degradation parameters
        is_train: Whether in training mode (enables stochastic degradations)
        
    Returns:
        Low-resolution PIL image
        
    Raises:
        RuntimeError: If degradation fails
    """
    if scale < 1:
        raise ValueError(f"Scale must be >= 1, got {scale}")
    
    if not isinstance(hr_image, Image.Image):
        raise TypeError(f"Image must be PIL Image, got {type(hr_image)}")
    
    degradation_config = degradation_config or {}
    image = hr_image.copy()

    try:
        # Apply blur if enabled
        if is_train and random.random() < float(degradation_config.get("blur_prob", 0.0)):
            radius_min = float(degradation_config.get("blur_radius_min", 0.2))
            radius_max = float(degradation_config.get("blur_radius_max", 1.2))
            blur_radius = random.uniform(radius_min, radius_max)
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Downsample
        method = _degradation_method(degradation_config, is_train)
        resample = PIL_RESAMPLING.get(method, Image.BICUBIC)

        w, h = image.size
        lr_w = max(1, w // scale)
        lr_h = max(1, h // scale)
        
        if lr_w == 0 or lr_h == 0:
            raise RuntimeError(f"Invalid LR size: {lr_w}x{lr_h} from {w}x{h} with scale {scale}")
        
        image = image.resize((lr_w, lr_h), resample=resample)

        tensor = TF.to_tensor(image)

        # Add noise if enabled
        if is_train and random.random() < float(degradation_config.get("noise_prob", 0.0)):
            noise_std_min = float(degradation_config.get("noise_std_min", 1.0)) / 255.0
            noise_std_max = float(degradation_config.get("noise_std_max", 8.0)) / 255.0
            noise_std = random.uniform(noise_std_min, noise_std_max)
            tensor = torch.clamp(tensor + torch.randn_like(tensor) * noise_std, 0.0, 1.0)
            image = TF.to_pil_image(tensor)

        # Apply JPEG compression if enabled
        if is_train and random.random() < float(degradation_config.get("jpeg_prob", 0.0)):
            quality_min = int(degradation_config.get("jpeg_quality_min", 40))
            quality_max = int(degradation_config.get("jpeg_quality_max", 90))
            jpeg_quality = random.randint(quality_min, quality_max)
            image = _apply_jpeg_compression(image.convert("RGB"), quality=jpeg_quality)
            if hr_image.mode == "L":
                image = image.convert("L")

        return image
        
    except Exception as e:
        raise RuntimeError(f"Degradation failed: {str(e)}") from e


class SuperResolutionDataset(Dataset):
    def __init__(
        self,
        hr_dir: str | Path,
        scale: int = 4,
        patch_size: int = 128,
        is_train: bool = True,
        srcnn_mode: bool = False,
        degradation_config: Dict[str, Any] | None = None,
        image_mode: str = "RGB",
    ) -> None:
        self.hr_paths = list_images(hr_dir)
        if not self.hr_paths:
            raise FileNotFoundError(f"No images found in {hr_dir}")
        self.scale = scale
        self.patch_size = patch_size
        self.is_train = is_train
        self.srcnn_mode = srcnn_mode
        self.degradation_config = degradation_config or {}
        self.image_mode = image_mode

    def __len__(self) -> int:
        return len(self.hr_paths)

    def _prepare_hr(self, hr_image: Image.Image) -> Image.Image:
        """
        Prepare high-resolution image for training/validation.
        
        Args:
            hr_image: PIL Image
            
        Returns:
            Prepared PIL image
            
        Raises:
            RuntimeError: If image is too small
        """
        width, height = hr_image.size
        
        # Validate image dimensions
        if width < self.scale or height < self.scale:
            raise RuntimeError(
                f"Image too small ({width}x{height}) for scale {self.scale}. "
                f"Minimum size: {self.scale}x{self.scale}"
            )
        
        # Crop to multiple of scale
        width = width - (width % self.scale)
        height = height - (height % self.scale)
        hr_image = hr_image.crop((0, 0, width, height))

        if self.is_train:
            hr_patch = paired_random_crop(hr_image, self.patch_size)
            hr_patch = augment_pair(hr_patch)
            return hr_patch

        eval_crop_size = int(self.degradation_config.get("eval_crop_size", 0) or 0)
        if eval_crop_size > 0 and min(width, height) > eval_crop_size:
            side = min(width, height, eval_crop_size)
            left = (width - side) // 2
            top = (height - side) // 2
            hr_image = hr_image.crop((left, top, left + side, top + side))
            side = side - (side % self.scale)
            
            if side < self.scale:
                raise RuntimeError(
                    f"Crop size {eval_crop_size} too small after alignment to scale {self.scale}"
                )
            
            hr_image = hr_image.crop((0, 0, side, side))
        
        return hr_image

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training/validation sample.
        
        Args:
            index: Sample index
            
        Returns:
            Tuple of (lr_tensor, hr_tensor) or (lr_up, hr_tensor) for SRCNN
            
        Raises:
            RuntimeError: If loading/processing fails
        """
        try:
            if index < 0 or index >= len(self.hr_paths):
                raise IndexError(f"Index {index} out of range [0, {len(self.hr_paths)})")
            
            img_path = self.hr_paths[index]
            
            try:
                hr_image = Image.open(img_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load image {img_path}: {str(e)}") from e
            
            # Validate image mode
            try:
                hr_image = hr_image.convert(self.image_mode)
            except Exception as e:
                raise RuntimeError(f"Failed to convert image mode to {self.image_mode}: {str(e)}") from e
            
            # Prepare HR image
            try:
                hr_image = self._prepare_hr(hr_image)
            except Exception as e:
                raise RuntimeError(f"Failed to prepare HR image from {img_path}: {str(e)}") from e
            
            # Create LR image
            try:
                lr_image = make_lr_from_hr(hr_image, self.scale, self.degradation_config, self.is_train)
            except Exception as e:
                raise RuntimeError(f"Failed to degrade HR image: {str(e)}") from e
            
            # Convert to tensors
            try:
                hr_tensor = TF.to_tensor(hr_image)
                lr_tensor = TF.to_tensor(lr_image)
            except Exception as e:
                raise RuntimeError(f"Failed to convert to tensor: {str(e)}") from e

            # SRCNN special case: upsampling
            if self.srcnn_mode:
                resize_method = str(self.degradation_config.get("srcnn_resize", "bicubic")).lower()
                try:
                    lr_up = TF.resize(
                        lr_tensor,
                        size=[hr_tensor.shape[1], hr_tensor.shape[2]],
                        interpolation=TORCH_INTERPOLATION.get(resize_method, InterpolationMode.BICUBIC),
                        antialias=True,
                    )
                    return lr_up, hr_tensor
                except Exception as e:
                    raise RuntimeError(f"SRCNN upsampling failed: {str(e)}") from e

            return lr_tensor, hr_tensor
            
        except Exception as e:
            raise RuntimeError(f"Dataset __getitem__ failed at index {index}: {str(e)}") from e
