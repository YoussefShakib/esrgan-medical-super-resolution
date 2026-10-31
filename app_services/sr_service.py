from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from models import build_model
from utils import get_device, load_config


ImageInput = Union[str, Path, Image.Image]

ProgressCallback = Callable[[int, int], None]  # (completed_tiles, total_tiles)


def _tile_positions(length: int, tile_size: int, stride: int) -> list[int]:
    """Tile start positions along one axis, clipped so the last tile ends at `length`."""
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    if positions[-1] != length - tile_size:
        positions.append(length - tile_size)
    return positions


def _ramp_weight(size: int, overlap_px: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """1-D linear feather: ramps up over `overlap_px`, plateau=1, ramps down over `overlap_px`."""
    if overlap_px <= 0 or size <= 2 * overlap_px:
        return torch.ones(size, device=device, dtype=dtype)
    ramp = torch.linspace(1.0 / (overlap_px + 1), 1.0, overlap_px, device=device, dtype=dtype)
    plateau = torch.ones(size - 2 * overlap_px, device=device, dtype=dtype)
    return torch.cat([ramp, plateau, ramp.flip(0)], dim=0)


@dataclass
class SuperResolutionService:
    config: dict
    device: torch.device
    model: torch.nn.Module
    config_path: str
    checkpoint_path: str

    def _forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(image_tensor).clamp(0, 1)

    def _forward_tiled(
        self,
        image_tensor: torch.Tensor,
        tile_size: int = 256,
        tile_overlap: int = 24,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> torch.Tensor:
        """Tiled forward with linear-ramp feather blending on overlaps (seamless)."""
        scale = int(self.config["scale"])
        _, channels, height, width = image_tensor.shape

        tile_size = max(32, int(tile_size))
        tile_overlap = max(0, min(int(tile_overlap), tile_size // 2))
        stride = max(1, tile_size - tile_overlap)

        ys = _tile_positions(height, tile_size, stride)
        xs = _tile_positions(width, tile_size, stride)
        total_tiles = len(ys) * len(xs)

        out_h = height * scale
        out_w = width * scale
        output = torch.zeros((1, channels, out_h, out_w), dtype=image_tensor.dtype, device=self.device)
        weight = torch.zeros_like(output)

        # Feather ramp matches the overlap region in the output (scaled) space.
        out_tile = tile_size * scale
        out_overlap = tile_overlap * scale
        ramp_y = _ramp_weight(out_tile, out_overlap, self.device, image_tensor.dtype)
        ramp_x = _ramp_weight(out_tile, out_overlap, self.device, image_tensor.dtype)
        feather = ramp_y[:, None] * ramp_x[None, :]

        completed = 0
        with torch.no_grad():
            for y in ys:
                for x in xs:
                    tile = image_tensor[:, :, y : y + tile_size, x : x + tile_size]
                    sr_tile = self.model(tile).clamp(0, 1)
                    sh, sw = sr_tile.shape[-2], sr_tile.shape[-1]
                    oy, ox = y * scale, x * scale
                    w_slice = feather[:sh, :sw]
                    output[:, :, oy : oy + sh, ox : ox + sw] += sr_tile * w_slice
                    weight[:, :, oy : oy + sh, ox : ox + sw] += w_slice
                    completed += 1
                    if progress_callback is not None:
                        try:
                            progress_callback(completed, total_tiles)
                        except Exception:
                            pass

        return output / weight.clamp_min(1e-8)

    def enhance_pil_image(
        self,
        image: Image.Image,
        use_tiling: bool = True,
        tile_size: int = 256,
        tile_overlap: int = 24,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Image.Image:
        """Super-resolve a PIL image. Uses ramp-blended tiling when `use_tiling` is True."""
        try:
            image = image.convert("RGB")
            image_tensor = TF.to_tensor(image).unsqueeze(0).to(self.device)

            if self.config["model_name"].lower() == "srcnn":
                width, height = image.size
                image_tensor = torch.nn.functional.interpolate(
                    image_tensor,
                    size=(height * self.config["scale"], width * self.config["scale"]),
                    mode="bicubic",
                    align_corners=False,
                )

            if use_tiling:
                sr_tensor = self._forward_tiled(
                    image_tensor=image_tensor,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    progress_callback=progress_callback,
                )
            else:
                sr_tensor = self._forward(image_tensor)
                if progress_callback is not None:
                    try:
                        progress_callback(1, 1)
                    except Exception:
                        pass

            result_image = TF.to_pil_image(sr_tensor.squeeze(0).cpu())

            del image_tensor, sr_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return result_image

        except Exception as e:
            raise RuntimeError(f"Image enhancement failed: {str(e)}") from e

    def enhance_image(
        self,
        image_input: ImageInput,
        use_tiling: bool = True,
        tile_size: int = 256,
        tile_overlap: int = 24,
    ) -> Image.Image:
        """
        Enhance an image from path or PIL Image.
        
        Args:
            image_input: Image path (str/Path) or PIL Image
            use_tiling: Whether to use tiling
            tile_size: Tile size
            tile_overlap: Tile overlap
            
        Returns:
            Enhanced PIL Image
            
        Raises:
            ValueError: If input is invalid
            FileNotFoundError: If image file not found
            RuntimeError: If enhancement fails
        """
        try:
            if isinstance(image_input, Image.Image):
                image = image_input
            elif isinstance(image_input, (str, Path)):
                image_path = Path(image_input)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                image = Image.open(image_path).convert("RGB")
            else:
                raise ValueError(f"Invalid image input type: {type(image_input)}")

            return self.enhance_pil_image(
                image=image,
                use_tiling=use_tiling,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
            )
            
        except (FileNotFoundError, ValueError):
            raise
        except Exception as e:
            raise RuntimeError(f"Image enhancement failed: {str(e)}") from e

    def enhance_and_save(
        self,
        image_input: ImageInput,
        output_path: Union[str, Path],
        use_tiling: bool = True,
        tile_size: int = 256,
        tile_overlap: int = 24,
    ) -> Path:
        """
        Enhance image and save to disk.
        
        Args:
            image_input: Image path or PIL Image
            output_path: Path to save enhanced image
            use_tiling: Whether to use tiling
            tile_size: Tile size
            tile_overlap: Tile overlap
            
        Returns:
            Path to saved image
            
        Raises:
            RuntimeError: If enhancement or save fails
        """
        try:
            sr_image = self.enhance_image(
                image_input=image_input,
                use_tiling=use_tiling,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
            )
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sr_image.save(output_path)
            return output_path
        except Exception as e:
            raise RuntimeError(f"Failed to enhance and save image: {str(e)}") from e


def load_sr_service(config_path: Union[str, Path], checkpoint_path: Union[str, Path]) -> SuperResolutionService:
    """
    Load super-resolution service with model and checkpoint.
    
    Args:
        config_path: Path to YAML config file
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Initialized SuperResolutionService
        
    Raises:
        FileNotFoundError: If config or checkpoint not found
        RuntimeError: If loading fails
    """
    try:
        config_path = Path(config_path)
        checkpoint_path = Path(checkpoint_path)
        
        # Validate files exist
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        config = load_config(str(config_path))
        device = get_device(config)

        model = build_model(config["model_name"], config["scale"]).to(device)
        
        # Load checkpoint with validation
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint (corrupted?): {str(e)}") from e
        
        if "model_state" not in checkpoint:
            raise RuntimeError("Checkpoint missing 'model_state' key")
        
        try:
            # Use strict=False to ignore extra keys, but log any missing keys
            incompatible_keys = model.load_state_dict(checkpoint["model_state"], strict=False)
            if incompatible_keys.missing_keys:
                import warnings
                warnings.warn(f"Missing model keys: {incompatible_keys.missing_keys}", RuntimeWarning)
        except Exception as e:
            raise RuntimeError(f"Failed to load model state: {str(e)}") from e
        
        model.eval()

        return SuperResolutionService(
            config=config,
            device=device,
            model=model,
            config_path=str(config_path),
            checkpoint_path=str(checkpoint_path),
        )
        
    except (FileNotFoundError, RuntimeError):
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load SR service: {str(e)}") from e