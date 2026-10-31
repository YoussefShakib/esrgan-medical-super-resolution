\
import csv
import json
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import yaml


def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Deprecated: Use src.core.load_config for validation.
    
    Args:
        path: Path to YAML config file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file not found
        RuntimeError: If YAML parsing fails
    """
    try:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise RuntimeError(f"Config file is empty: {path}")
        
        return config
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {str(e)}") from e


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set CUDA determinism for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path (str or Path)
        
    Returns:
        Path object of the created directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device(config: Dict[str, Any]) -> torch.device:
    """
    Get torch device based on config.
    
    Deprecated: Use src.core.DeviceManager instead for better configuration.
    
    Args:
        config: Configuration dictionary with 'device' key
        
    Returns:
        torch.device object
    """
    requested = str(config.get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CSVLogger:
    """Logger for writing training metrics to CSV file."""
    
    def __init__(self, csv_path: str | Path, fieldnames: list[str]) -> None:
        """
        Initialize CSV logger.
        
        Args:
            csv_path: Path to CSV file
            fieldnames: List of column names
        """
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self._initialized = self.csv_path.exists()

    def write(self, row: Dict[str, Any]) -> None:
        """
        Write a row to CSV file.
        
        Args:
            row: Dictionary with values for each fieldname
            
        Raises:
            ValueError: If row has missing or extra fields
        """
        # Validate row
        missing_fields = set(self.fieldnames) - set(row.keys())
        if missing_fields:
            raise ValueError(f"Missing fields in CSV row: {missing_fields}")
        
        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                if not self._initialized:
                    writer.writeheader()
                    self._initialized = True
                writer.writerow(row)
        except Exception as e:
            raise RuntimeError(f"Failed to write to CSV {self.csv_path}: {str(e)}") from e


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        path: Output JSON file path
        data: Dictionary to save
        
    Raises:
        RuntimeError: If save fails
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to save JSON to {path}: {str(e)}") from e


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
