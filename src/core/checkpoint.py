"""Checkpoint management with atomic saves."""

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .exceptions import CheckpointError
from .logging_setup import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Manages model checkpoint saving and loading."""

    @staticmethod
    def save_atomic(
        path: Path,
        state_dict: Dict[str, Any],
        description: str = "",
    ) -> None:
        """
        Save checkpoint atomically to prevent corruption.
        
        Args:
            path: Output path for checkpoint
            state_dict: State dictionary to save
            description: Optional description of checkpoint
            
        Raises:
            CheckpointError: If save fails
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to temporary file first
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                suffix=".pth",
                delete=False
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
            
            try:
                # Save state to temporary file
                torch.save(state_dict, tmp_path)
                
                # Atomic rename
                tmp_path.replace(path)
                
                logger.debug(f"Checkpoint saved atomically: {path} {description}")
                
            except Exception as e:
                # Clean up temp file on failure
                if tmp_path.exists():
                    tmp_path.unlink()
                raise
                
        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint {path}: {str(e)}") from e

    @staticmethod
    def load(
        path: Path,
        device: torch.device,
        strict: bool = True,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Load checkpoint with validation.
        
        Args:
            path: Path to checkpoint
            device: Device to load on
            strict: Whether to require exact state dict match
            validate: Whether to validate checkpoint integrity
            
        Returns:
            Loaded state dictionary
            
        Raises:
            CheckpointError: If load fails or checkpoint is invalid
        """
        try:
            path = Path(path)
            if not path.exists():
                raise CheckpointError(f"Checkpoint not found: {path}")
            
            # Try to load
            try:
                checkpoint = torch.load(path, map_location=device)
            except Exception as e:
                raise CheckpointError(f"Failed to load checkpoint (corrupted?): {str(e)}") from e
            
            # Validate structure
            if validate:
                if not isinstance(checkpoint, dict):
                    raise CheckpointError(f"Checkpoint must be dict, got {type(checkpoint)}")
                if "model_state" not in checkpoint:
                    raise CheckpointError("Checkpoint missing 'model_state' key")
            
            logger.debug(f"Checkpoint loaded: {path}")
            return checkpoint
            
        except CheckpointError:
            raise
        except Exception as e:
            raise CheckpointError(f"Unexpected error loading checkpoint: {str(e)}") from e

    @staticmethod
    def load_model_state(
        model: torch.nn.Module,
        path: Path,
        device: torch.device,
        strict: bool = False,
    ) -> Optional[int]:
        """
        Load model state dict from checkpoint.
        
        Args:
            model: Model to load into
            path: Path to checkpoint
            device: Device to load on
            strict: Strict mode for state dict loading
            
        Returns:
            Epoch number if available, otherwise None
            
        Raises:
            CheckpointError: If load fails
        """
        try:
            checkpoint = CheckpointManager.load(path, device, strict=strict, validate=True)
            
            # Validate model state has required structure
            model_state = checkpoint.get("model_state")
            if model_state is None:
                raise CheckpointError("Checkpoint missing 'model_state'")
            
            # Load state dict
            try:
                model.load_state_dict(model_state, strict=strict)
            except RuntimeError as e:
                if strict:
                    raise CheckpointError(f"Strict load failed: {str(e)}") from e
                else:
                    logger.warning(f"Non-strict load: {str(e)}")
            
            epoch = checkpoint.get("epoch")
            logger.info(f"Model state loaded from {path} (epoch {epoch})" if epoch else f"Model state loaded from {path}")
            
            return epoch
            
        except CheckpointError:
            raise
        except Exception as e:
            raise CheckpointError(f"Unexpected error loading model state: {str(e)}") from e

    @staticmethod
    def load_training_state(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        path: Path,
        device: torch.device,
    ) -> int:
        """
        Load complete training state (model + optimizer).
        
        Args:
            model: Model to load into
            optimizer: Optimizer to load into
            path: Path to checkpoint
            device: Device to load on
            
        Returns:
            Next epoch to resume from
            
        Raises:
            CheckpointError: If load fails
        """
        try:
            checkpoint = CheckpointManager.load(path, device, strict=False, validate=True)
            
            # Load model
            model.load_state_dict(checkpoint["model_state"], strict=False)
            
            # Load optimizer if available
            if "optimizer_state" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
                except Exception as e:
                    logger.warning(f"Could not load optimizer state: {e}")
            
            epoch = checkpoint.get("epoch", 0)
            logger.info(f"Training state loaded from {path}, resuming from epoch {epoch + 1}")
            
            return int(epoch) + 1
            
        except CheckpointError:
            raise
        except Exception as e:
            raise CheckpointError(f"Failed to load training state: {str(e)}") from e
