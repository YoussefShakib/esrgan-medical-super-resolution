"""Comprehensive test suite for SR project."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# Add project root and src/ to path
import sys
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from core.config import load_config, Config, validate_config
from core.logging_setup import setup_logging
from core.device import DeviceManager
from core.checkpoint import CheckpointManager
from core.exceptions import InvalidConfigError, CheckpointError
from dataset import SuperResolutionDataset, make_lr_from_hr
from models import build_model
from utils import count_parameters, save_json, set_seed
from metrics import calculate_psnr, calculate_ssim


class TestConfig:
    """Test configuration management."""
    
    def test_load_valid_config(self, tmp_path):
        """Test loading valid config file."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
experiment_name: test
model_name: srcnn
scale: 4
seed: 42
device: auto
save_dir: runs

dataset:
  train_hr_dir: datasets/train_HR
  val_hr_dir: datasets/val_HR
  patch_size: 64

training:
  batch_size: 1
  num_workers: 0
  epochs: 1
  lr: 0.0001
  lr_step: 10
  lr_gamma: 0.5
""")
        
        config = load_config(str(config_file))
        assert config.model_name == "srcnn"
        assert config.scale == 4
        assert config.training.batch_size == 1

    def test_load_invalid_config(self, tmp_path):
        """Test loading invalid config raises error."""
        config_file = tmp_path / "invalid_config.yaml"
        config_file.write_text("""
experiment_name: test
model_name: invalid_model
scale: 4
dataset:
  train_hr_dir: datasets/train_HR
  val_hr_dir: datasets/val_HR
  patch_size: 64
training:
  batch_size: 1
training_epochs: 1
""")
        
        with pytest.raises(InvalidConfigError):
            load_config(str(config_file))

    def test_invalid_batch_size(self, tmp_path):
        """Test invalid batch size is caught."""
        config_file = tmp_path / "bad_batch.yaml"
        config_file.write_text("""
experiment_name: test
model_name: srcnn
scale: 4
dataset:
  train_hr_dir: datasets/train_HR
  val_hr_dir: datasets/val_HR
  patch_size: 64
training:
  batch_size: 0
  num_workers: 0
  epochs: 1
  lr: 0.0001
  lr_step: 10
  lr_gamma: 0.5
""")
        
        with pytest.raises(InvalidConfigError):
            load_config(str(config_file))


class TestDataset:
    """Test dataset functionality."""
    
    @pytest.fixture
    def sample_image_dataset(self, tmp_path):
        """Create sample dataset with test images."""
        train_dir = tmp_path / "train_HR"
        val_dir = tmp_path / "val_HR"
        train_dir.mkdir()
        val_dir.mkdir()
        
        # Create test images
        for i in range(3):
            img = Image.new("RGB", (256, 256), color=(i*50, i*50, i*50))
            img.save(train_dir / f"train_{i}.png")
        
        for i in range(1):
            img = Image.new("RGB", (256, 256), color=(100, 100, 100))
            img.save(val_dir / f"val_{i}.png")
        
        return train_dir, val_dir

    def test_dataset_loading(self, sample_image_dataset):
        """Test dataset loads images correctly."""
        train_dir, val_dir = sample_image_dataset
        dataset = SuperResolutionDataset(
            hr_dir=train_dir,
            scale=4,
            patch_size=64,
            is_train=True,
        )
        
        assert len(dataset) == 3
        lr, hr = dataset[0]
        # LR is downsampled by scale=4, so 64÷4 = 16
        assert lr.shape == (3, 16, 16)
        assert hr.shape == (3, 64, 64)

    def test_degradation(self, sample_image_dataset):
        """Test image degradation."""
        train_dir, _ = sample_image_dataset
        dataset = SuperResolutionDataset(
            hr_dir=train_dir,
            scale=4,
            patch_size=128,
            is_train=True,
            degradation_config={
                "blur_prob": 1.0,
                "blur_radius_min": 0.5,
                "blur_radius_max": 1.0,
                "noise_prob": 1.0,
            },
        )
        
        lr, hr = dataset[0]
        assert torch.isfinite(lr).all()
        assert torch.isfinite(hr).all()

    def test_make_lr_from_hr(self, sample_image_dataset):
        """Test LR creation from HR."""
        train_dir, _ = sample_image_dataset
        img = Image.open(list(train_dir.glob("*.png"))[0])
        
        lr = make_lr_from_hr(img, scale=4, is_train=False)
        assert lr.size == (img.width // 4, img.height // 4)


class TestModels:
    """Test model architectures."""
    
    @pytest.mark.parametrize("model_name", ["srcnn", "srresnet", "edsr"])
    def test_model_forward(self, model_name):
        """Test model forward pass."""
        model = build_model(model_name, scale=4)
        model.eval()
        
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        
        assert y.shape[0] == 1
        assert y.shape[1] == 3
        if model_name == "srcnn":
            assert y.shape[2] == 64 and y.shape[3] == 64
        else:
            assert y.shape[2] == 256 and y.shape[3] == 256

    def test_parameter_count(self):
        """Test parameter counting."""
        model = build_model("srcnn", scale=4)
        params = count_parameters(model)
        assert params > 0

    def test_invalid_scale(self):
        """Test invalid scale raises error."""
        with pytest.raises(ValueError):
            build_model("edsr", scale=3)


class TestCheckpoint:
    """Test checkpoint management."""
    
    def test_save_and_load_atomic(self, tmp_path):
        """Test atomic checkpoint save and load."""
        model = build_model("srcnn", scale=4)
        
        state = {"model_state": model.state_dict(), "epoch": 1}
        checkpoint_path = tmp_path / "test.pth"
        
        CheckpointManager.save_atomic(checkpoint_path, state)
        assert checkpoint_path.exists()
        
        loaded = CheckpointManager.load(checkpoint_path, torch.device("cpu"))
        assert loaded["epoch"] == 1

    def test_load_model_state(self, tmp_path):
        """Test loading model state."""
        model1 = build_model("srcnn", scale=4)
        model2 = build_model("srcnn", scale=4)
        
        state = {"model_state": model1.state_dict(), "epoch": 5}
        checkpoint_path = tmp_path / "model.pth"
        CheckpointManager.save_atomic(checkpoint_path, state)
        
        epoch = CheckpointManager.load_model_state(
            model2, checkpoint_path, torch.device("cpu"), strict=True
        )
        assert epoch == 5


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_psnr_calculation(self):
        """Test PSNR calculation."""
        sr = torch.ones(1, 3, 64, 64)
        hr = torch.ones(1, 3, 64, 64)
        
        psnr = calculate_psnr(sr, hr)
        assert psnr > 0  # Identical images should have high PSNR

    def test_ssim_calculation(self):
        """Test SSIM calculation."""
        sr = torch.ones(1, 3, 64, 64)
        hr = torch.ones(1, 3, 64, 64)
        
        ssim = calculate_ssim(sr, hr)
        assert 0 <= ssim <= 1
        assert abs(ssim - 1.0) < 0.01  # Identical images should have SSIM close to 1


class TestLogging:
    """Test logging setup."""
    
    def test_setup_logging(self, tmp_path):
        """Test logging initialization."""
        logger = setup_logging(tmp_path, name="test_logger")
        assert logger is not None
        
        log_file = tmp_path / "test_logger.log"
        assert log_file.exists()


class TestDevice:
    """Test device management."""
    
    def test_device_setup(self):
        """Test device setup."""
        config = {"device": "cpu", "seed": 42}
        device = DeviceManager.setup(config)
        assert device.type == "cpu"

    def test_device_cleanup(self):
        """Test device cleanup."""
        DeviceManager.cleanup()  # Should not raise


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting for reproducibility."""
        set_seed(42)
        a = np.random.rand(5)
        
        set_seed(42)
        b = np.random.rand(5)
        
        assert np.allclose(a, b)

    def test_save_json(self, tmp_path):
        """Test JSON saving."""
        data = {"key": "value", "number": 42}
        file_path = tmp_path / "test.json"
        
        save_json(file_path, data)
        assert file_path.exists()


class TestLPIPS:
    """Test LPIPS metric."""

    def test_lpips_identical(self):
        """Identical images should have LPIPS ~ 0."""
        from metrics import calculate_lpips
        t = torch.ones(1, 3, 64, 64) * 0.5
        score = calculate_lpips(t, t)
        assert 0.0 <= score < 0.05, f"LPIPS of identical images should be ~0, got {score}"

    def test_lpips_different(self):
        """Very different images should have high LPIPS."""
        from metrics import calculate_lpips
        zeros = torch.zeros(1, 3, 64, 64)
        ones = torch.ones(1, 3, 64, 64)
        score = calculate_lpips(zeros, ones)
        assert score > 0.3, f"LPIPS of all-zero vs all-one should be >0.3, got {score}"

    def test_batch_metrics_with_lpips(self):
        """batch_metrics returns lpips key when compute_lpips=True."""
        from metrics import batch_metrics
        sr = torch.rand(1, 3, 64, 64)
        hr = torch.rand(1, 3, 64, 64)
        result = batch_metrics(sr, hr, compute_lpips=True)
        assert "lpips" in result
        assert 0.0 <= result["lpips"] <= 2.0

    def test_batch_metrics_without_lpips(self):
        """batch_metrics does NOT return lpips key when compute_lpips=False."""
        from metrics import batch_metrics
        sr = torch.rand(1, 3, 64, 64)
        hr = torch.rand(1, 3, 64, 64)
        result = batch_metrics(sr, hr, compute_lpips=False)
        assert "lpips" not in result
        assert "psnr" in result and "ssim" in result


class TestStableGANConfig:
    """Test that stable-GAN config fields load and validate correctly."""

    def test_stable_gan_defaults(self, tmp_path):
        """stable_gan fields have correct defaults when absent from YAML."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""
experiment_name: test_stable
model_name: esrgan
scale: 4
seed: 42
device: auto
save_dir: runs

dataset:
  train_hr_dir: datasets/train_HR
  val_hr_dir: datasets/val_HR
  patch_size: 64

training:
  batch_size: 1
  num_workers: 0
  epochs: 1
  lr_g: 0.0001
  lr_d: 0.0001
  gan_start_epoch: 2
  pixel_weight: 1.0
  perceptual_weight: 0.01
  adv_weight: 0.005
""")
        config = load_config(str(config_file))
        assert config.training.stable_gan is False
        assert config.training.disc_warmup_epochs == 2
        assert config.training.r1_penalty_weight == 10.0

    def test_stable_gan_enabled(self, tmp_path):
        """stable_gan=true parses correctly with custom values."""
        config_file = tmp_path / "test_stable.yaml"
        config_file.write_text("""
experiment_name: test_stable_on
model_name: esrgan
scale: 4
seed: 42
device: auto
save_dir: runs

dataset:
  train_hr_dir: datasets/train_HR
  val_hr_dir: datasets/val_HR
  patch_size: 64

training:
  batch_size: 1
  num_workers: 0
  epochs: 1
  lr_g: 0.0001
  lr_d: 0.0001
  gan_start_epoch: 5
  pixel_weight: 1.0
  perceptual_weight: 0.01
  adv_weight: 0.005
  stable_gan: true
  disc_warmup_epochs: 3
  r1_penalty_weight: 5.0
""")
        config = load_config(str(config_file))
        assert config.training.stable_gan is True
        assert config.training.disc_warmup_epochs == 3
        assert config.training.r1_penalty_weight == 5.0


class TestStratifiedSplit:
    """Test the stratified BACH split logic."""

    def test_split_balance(self, tmp_path):
        """Rebalance script produces exactly VAL_PER_CLASS images per class."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from rebalance_split import build_split, CLASSES, VAL_PER_CLASS, TRAIN_PER_CLASS

        flat_dir = tmp_path / "bach_flat"
        flat_dir.mkdir()
        # Create 100 dummy images per class
        for cls in CLASSES:
            for i in range(1, 101):
                (flat_dir / f"{cls}_{cls[0].lower()}{i:03d}.tif").touch()

        split = build_split(flat_dir)
        for cls in CLASSES:
            assert len(split["val"][cls]) == VAL_PER_CLASS, \
                f"Expected {VAL_PER_CLASS} val images for {cls}"
            assert len(split["train"][cls]) == TRAIN_PER_CLASS, \
                f"Expected {TRAIN_PER_CLASS} train images for {cls}"

    def test_split_no_overlap(self, tmp_path):
        """No image should appear in both train and val."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from rebalance_split import build_split, CLASSES

        flat_dir = tmp_path / "bach_flat"
        flat_dir.mkdir()
        for cls in CLASSES:
            for i in range(1, 101):
                (flat_dir / f"{cls}_{cls[0].lower()}{i:03d}.tif").touch()

        split = build_split(flat_dir)
        for cls in CLASSES:
            train_names = {p.name for p in split["train"][cls]}
            val_names = {p.name for p in split["val"][cls]}
            overlap = train_names & val_names
            assert len(overlap) == 0, f"Overlap in class {cls}: {overlap}"

    def test_split_deterministic(self, tmp_path):
        """Same input always produces the same split."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from rebalance_split import build_split, CLASSES

        flat_dir = tmp_path / "bach_flat"
        flat_dir.mkdir()
        for cls in CLASSES:
            for i in range(1, 101):
                (flat_dir / f"{cls}_{cls[0].lower()}{i:03d}.tif").touch()

        split_a = build_split(flat_dir)
        split_b = build_split(flat_dir)
        for cls in CLASSES:
            assert [p.name for p in split_a["val"][cls]] == [p.name for p in split_b["val"][cls]]


class TestAblationConfigGeneration:
    """Test that ablation configs are valid and have the expected differences."""

    def test_ablation_bicubic_config(self):
        """esrgan_ablation_bicubic.yaml loads and has bicubic-only degradation."""
        config = load_config("configs/esrgan_ablation_bicubic.yaml")
        assert config.model_name == "esrgan"
        assert config.dataset.degradation.get("method") == "bicubic"
        assert config.training.stable_gan is True

    def test_ablation_mixed_config(self):
        """esrgan_ablation_mixed.yaml loads and has mixed degradation."""
        config = load_config("configs/esrgan_ablation_mixed.yaml")
        assert config.model_name == "esrgan"
        assert config.dataset.degradation.get("method") == "mixed"
        assert config.training.stable_gan is True

    def test_configs_differ_only_in_degradation(self):
        """Both ablation configs are identical except for the degradation pipeline."""
        cfg_bic = load_config("configs/esrgan_ablation_bicubic.yaml")
        cfg_mix = load_config("configs/esrgan_ablation_mixed.yaml")
        # Hyperparameters must match
        assert cfg_bic.training.lr_g == cfg_mix.training.lr_g
        assert cfg_bic.training.epochs == cfg_mix.training.epochs
        assert cfg_bic.training.gan_start_epoch == cfg_mix.training.gan_start_epoch
        assert cfg_bic.training.stable_gan == cfg_mix.training.stable_gan
        # But degradation method must differ
        assert cfg_bic.dataset.degradation.get("method") != cfg_mix.dataset.degradation.get("method")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
