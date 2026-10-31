"""Device management for SR project."""

from typing import Any, Dict

import torch

from .exceptions import DeviceError
from .logging_setup import get_logger

logger = get_logger(__name__)


class DeviceManager:
    """Manages device setup and CUDA configuration."""

    @staticmethod
    def setup(config: Dict[str, Any]) -> torch.device:
        """
        Setup device with proper CUDA configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            torch.device object
            
        Raises:
            DeviceError: If device setup fails
        """
        try:
            requested_device = str(config.get("device", "auto")).lower()
            
            if requested_device == "auto":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif requested_device == "cuda":
                if not torch.cuda.is_available():
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    device = torch.device("cpu")
                else:
                    device = torch.device("cuda")
            elif requested_device == "cpu":
                device = torch.device("cpu")
            else:
                raise DeviceError(f"Unknown device: {requested_device}")
            
            # Set CUDA determinism for reproducibility
            if device.type == "cuda":
                torch.cuda.manual_seed_all(config.get("seed", 42))
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.cuda.empty_cache()
                
                # Log GPU info
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
            else:
                torch.backends.cudnn.enabled = False
                logger.info("Using CPU device")
            
            logger.info(f"Device: {device}")
            return device
            
        except Exception as e:
            raise DeviceError(f"Failed to setup device: {str(e)}") from e

    @staticmethod
    def cleanup() -> None:
        """Clean up GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU memory cleared")
