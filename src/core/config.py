"""Configuration management and validation."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .exceptions import InvalidConfigError
from .logging_setup import get_logger

logger = get_logger(__name__)


class DatasetConfig(BaseModel):
    """Dataset configuration."""
    train_hr_dir: str
    val_hr_dir: str
    patch_size: int = Field(ge=32, le=512)
    eval_crop_size: int = Field(default=512, ge=0)
    image_mode: str = Field(default="RGB", pattern="^(RGB|L|RGBA)$")
    degradation: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("train_hr_dir", "val_hr_dir")
    @classmethod
    def validate_dirs_exist(cls, v: str) -> str:
        path = Path(v)
        if not path.exists():
            logger.warning(f"Dataset directory does not exist yet: {v}")
        return v

    model_config = ConfigDict(extra="allow")


class TrainingConfig(BaseModel):
    """Training configuration."""
    batch_size: int = Field(ge=1, le=256)
    num_workers: int = Field(default=0, ge=0, le=16)
    persistent_workers: bool = Field(default=False)
    pin_memory: Optional[bool] = None
    epochs: int = Field(ge=1, le=10000)
    lr: Optional[float] = Field(default=None, ge=1e-7, le=1e-2)
    lr_g: Optional[float] = Field(default=None, ge=1e-7, le=1e-2)
    lr_d: Optional[float] = Field(default=None, ge=1e-7, le=1e-2)
    lr_step: int = Field(default=20, ge=1)
    lr_gamma: float = Field(default=0.5, ge=0.01, le=1.0)
    resume: bool = Field(default=True)
    gan_start_epoch: int = Field(default=10, ge=0)
    pixel_weight: float = Field(default=1.0, ge=0.0)
    perceptual_weight: float = Field(default=0.01, ge=0.0)
    adv_weight: float = Field(default=0.005, ge=0.0)
    pretrained_generator: Optional[str] = None
    use_mixed_precision: bool = Field(default=False)
    gradient_clip_value: Optional[float] = Field(default=None, ge=0.0)
    early_stopping_patience: int = Field(default=0, ge=0)
    eval_every_epochs: int = Field(default=1, ge=1)
    sample_every_epochs: int = Field(default=1, ge=1)
    max_val_images: int = Field(default=0, ge=0)
    # Stable-GAN schedule (opt-in via --stable-gan or stable_gan: true in config)
    stable_gan: bool = Field(default=False)
    disc_warmup_epochs: int = Field(default=2, ge=0)
    r1_penalty_weight: float = Field(default=10.0, ge=0.0)

    model_config = ConfigDict(extra="allow")


class Config(BaseModel):
    """Main configuration model."""
    experiment_name: str
    model_name: str = Field(pattern="^(srcnn|srresnet|edsr|esrgan)$")
    scale: int = Field(ge=1, le=8)
    seed: int = Field(default=42, ge=0)
    device: str = Field(default="auto", pattern="^(auto|cuda|cpu)$")
    save_dir: str = Field(default="runs")
    dataset: DatasetConfig
    training: TrainingConfig
    config_path: Optional[str] = None

    @field_validator("model_name", mode="before")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        valid_models = {"srcnn", "srresnet", "edsr", "esrgan"}
        if v.lower() not in valid_models:
            raise ValueError(f"Model must be one of {valid_models}, got {v}")
        return v.lower()

    @field_validator("scale", mode="before")
    @classmethod
    def validate_scale(cls, v: int) -> int:
        if v not in {2, 4, 8}:
            logger.warning(f"Scale {v} not in recommended values (2, 4, 8)")
        return v

    model_config = ConfigDict(extra="allow", protected_namespaces=())


def load_config(config_path: str) -> Config:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Validated Config object
        
    Raises:
        InvalidConfigError: If config is invalid
    """
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            raise InvalidConfigError(f"Config file not found: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict is None:
            raise InvalidConfigError(f"Config file is empty: {config_path}")
        
        # Store config path for later reference
        config_dict["config_path"] = str(config_path)
        
        # Validate with Pydantic
        config = Config(**config_dict)
        logger.info(f"Config loaded and validated: {config_path}")
        return config
        
    except InvalidConfigError:
        raise
    except Exception as e:
        raise InvalidConfigError(f"Failed to load/validate config: {str(e)}") from e


def validate_config(config_dict: Dict[str, Any]) -> Config:
    """
    Validate a configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Validated Config object
        
    Raises:
        InvalidConfigError: If config is invalid
    """
    try:
        return Config(**config_dict)
    except Exception as e:
        raise InvalidConfigError(f"Invalid configuration: {str(e)}") from e
