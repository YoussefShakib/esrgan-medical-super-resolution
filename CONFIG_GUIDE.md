# Configuration Guide

## Overview

Configurations are YAML files that define training parameters. All config files must validate against the schema to ensure correctness.

## Config Structure

```yaml
# Model configuration
experiment_name: my_experiment  # Experiment identifier
model_name: srcnn              # Model: srcnn, srresnet, edsr, esrgan
scale: 4                        # Upsampling scale: 2, 4, or 8
seed: 42                        # Random seed for reproducibility
device: auto                    # auto, cuda, or cpu
save_dir: runs                  # Output directory

# Dataset configuration
dataset:
  train_hr_dir: datasets/train_HR     # Training HR images
  val_hr_dir: datasets/val_HR         # Validation HR images
  patch_size: 128                     # Training patch size (32-512)
  eval_crop_size: 512                 # Eval crop size (0=disabled)
  image_mode: RGB                     # RGB or L (grayscale)
  degradation:                        # Optional degradations
    method: bicubic                   # bicubic, bilinear, nearest, lanczos, mixed
    mixed_methods: [bicubic, bilinear]# For mixed method
    blur_prob: 0.1                    # Probability of blur (0-1)
    blur_radius_min: 0.2
    blur_radius_max: 1.2
    noise_prob: 0.1                   # Gaussian noise
    noise_std_min: 1.0
    noise_std_max: 8.0
    jpeg_prob: 0.1                    # JPEG compression
    jpeg_quality_min: 40
    jpeg_quality_max: 90

# Training configuration
training:
  batch_size: 4                       # Batch size (1-256)
  num_workers: 0                      # Data loading workers (0-16)
  epochs: 100                         # Number of epochs
  
  # Basic model learning  
  lr: 0.0001                          # Learning rate (1e-7 to 1e-2)
  lr_step: 20                         # Scheduler step size
  lr_gamma: 0.5                       # LR decay factor
  
  # ESRGAN-specific learning
  lr_g: 0.0001                        # Generator LR
  lr_d: 0.0001                        # Discriminator LR
  
  # ESRGAN training control
  gan_start_epoch: 10                 # When to start adversarial training
  pixel_weight: 1.0                   # L1/Charbonnier loss weight
  perceptual_weight: 0.01             # VGG perceptual loss weight
  adv_weight: 0.005                   # Adversarial loss weight
  pretrained_generator: null           # Path to pretrained generator (optional)
  
  # Training advanced options
  use_mixed_precision: false          # Mixed precision training (experimental)
  gradient_clip_value: null           # Gradient clipping (null=disable)
  early_stopping_patience: 0          # Early stopping (0=disable)
  resume: true                        # Resume from last checkpoint
```

## Validation Rules

- `experiment_name`: Non-empty string
- `model_name`: Must be one of: srcnn, srresnet, edsr, esrgan
- `scale`: Must be 2, 4, or 8
- `batch_size`: 1-256
- `patch_size`: 32-512
- `num_workers`: 0-16 (0 on Windows recommended)
- `lr`: 1e-7 to 1e-2
- `epochs`: 1 to 10,000

## Examples

### Minimal SRResNet Config

```yaml
experiment_name: srresnet_demo
model_name: srresnet
scale: 4
seed: 42
device: auto
save_dir: runs

dataset:
  train_hr_dir: datasets/train_HR
  val_hr_dir: datasets/val_HR
  patch_size: 64

training:
  batch_size: 4
  num_workers: 0
  epochs: 50
  lr: 0.0001
  lr_step: 10
  lr_gamma: 0.5
```

### Full ESRGAN Config with All Features

```yaml
experiment_name: esrgan_full
model_name: esrgan
scale: 4
seed: 42
device: auto
save_dir: runs

dataset:
  train_hr_dir: datasets/train_HR
  val_hr_dir: datasets/val_HR
  patch_size: 192
  eval_crop_size: 512
  image_mode: RGB
  degradation:
    method: mixed
    mixed_methods: [bicubic, bilinear, nearest]
    blur_prob: 0.25
    blur_radius_min: 0.2
    blur_radius_max: 1.2
    noise_prob: 0.1
    noise_std_min: 1.0
    noise_std_max: 8.0
    jpeg_prob: 0.1
    jpeg_quality_min: 40
    jpeg_quality_max: 90

training:
  batch_size: 8
  num_workers: 4
  epochs: 200
  
  lr_g: 0.0001
  lr_d: 0.0001
  lr_step: 50
  lr_gamma: 0.5
  
  gan_start_epoch: 20
  pixel_weight: 1.0
  perceptual_weight: 0.01
  adv_weight: 0.005
  
  gradient_clip_value: 1.0
  early_stopping_patience: 20
  resume: true
```

## Configuration Parameters Explained

### Dataset

- **train_hr_dir/val_hr_dir**: Must contain only HR images. LR is created on-the-fly.
- **patch_size**: Larger = longer training, requires more memory. Typical: 64-192.
- **eval_crop_size**: Crop validation images to this size. Set to 0 to disable.
- **image_mode**: RGB for color, L for grayscale.
- **degradation**: Simulate realistic degradations during training.

### Training

- **batch_size**: Larger = faster training but more memory. Typical: 4-16.
- **num_workers**: Parallel data loading. Set to 0 on Windows.
- **lr**: Learning rate. Too high = divergence, too low = slow convergence.
- **lr_step/lr_gamma**: Reduce LR by `gamma` every `step` epochs.
- **gan_start_epoch**: Start adversarial training after warmup with pixel + perceptual loss.
- **weights**: Control loss components (pixel + perceptual + adversarial).
- **early_stopping_patience**: Stop if no improvement for N epochs. Set to 0 to disable.

## Tips for Configuration

1. **Start simple**: Use basic config first, add complexity gradually.
2. **Memory-limited?**: Reduce `batch_size`, `patch_size`, `num_workers`.
3. **Slow training?**: Increase `batch_size`, `num_workers`.
4. **Model diverging?**: Reduce `lr`, enable `gradient_clip_value`.
5. **Need fast convergence?**: Use larger `batch_size`, bigger dataset, better `lr_gamma`.
6. **For medical images**: Consider lower learning rates, more degradations.

## Common Mistakes

❌ `num_workers > 0` on Windows (hangs)  
✅ Always use `num_workers: 0` on Windows

❌ Storing checkpoints in slow directory  
✅ Use fast SSD for `save_dir`

❌ Very large `patch_size` with large batch  
✅ Balance both based on available memory

❌ Validating after every batch  
✅ Validation only after each epoch
