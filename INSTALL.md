# Installation & Setup Guide

## Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration, optional)
- 4GB RAM minimum (8GB+ recommended)

## Step 1: Create Virtual Environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Step 2: Install PyTorch

Visit [pytorch.org](https://pytorch.org) and get the install command for your system.

Example for Windows with CUDA 11.8:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### 1. Download a Sample Dataset

For demo, create minimal dataset:

```bash
mkdir -p datasets/DEMO/train_HR datasets/DEMO/val_HR
```

Add 10-15 HR images to `datasets/DEMO/train_HR` and 1-2 to `datasets/DEMO/val_HR`.

### 2. Train a Model

```bash
python src/train.py --config configs/srcnn_demo.yaml
```

### 3. Monitor Results

```bash
python src/plot_logs.py --csv runs/srcnn_demo_x4/train_log.csv
```

## Troubleshooting

### Issue: CUDA out of memory

**Solution**: Reduce `batch_size` in config:
```yaml
training:
  batch_size: 2  # Instead of 4
```

### Issue: No module named 'core'

**Solution**: Ensure you're running from the project root:
```bash
cd /path/to/esr_med
python src/train.py --config configs/srcnn_demo.yaml
```

### Issue: Image files not loading

**Solution**: Verify image format:
- Supported: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.webp`
- Files must be RGB or grayscale

### Issue: Out of memory during data loading

**Solution**: Reduce `num_workers` in config:
```yaml
training:
  num_workers: 0  # Or reduce from current value
```

## Project Structure

```
esr_med/
├── src/
│   ├── core/          # Core utilities (logging, config, checkpoint, device)
│   ├── models/        # Model architectures
│   ├── train.py       # Main training script
│   ├── infer.py       # Inference script
│   ├── evaluate.py    # Evaluation script
│   └── ...
├── app_services/      # Application services (SR, analysis, visuals)
├── configs/          # YAML configuration files
├── datasets/         # Dataset directory
├── runs/            # Training outputs
└── requirements.txt
```

## Running the Demo App

```bash
streamlit run app.py
```

The app launches with locked-in demo defaults — no flags or extra setup needed:

| Setting | Value | Source |
|---|---|---|
| BACH checkpoint | `runs/esrgan_ablation_mixed/checkpoints/best_ssim_generator.pth` | `scripts/gen_defense_figures.py` "Stable-ESRGAN" registry entry |
| BACH config | `configs/esrgan_ablation_mixed.yaml` | same |
| Brain MRI checkpoint | `runs/esrgan_brain_mri_x4/checkpoints/best_generator.pth` (epoch 10, PSNR 33.60) | `scripts/gen_defense_figures.py` (the `best_ssim_generator.pth` for this run is stale, re-saved at the GAN-start epoch) |
| Brain MRI config | `configs/esrgan_brain_mri.yaml` | — |
| Scale | ×4 | both configs |
| Precision | fp32 | no `.half()` calls anywhere in the inference path |
| Device | auto (GPU if available, else CPU) | `device: auto` in both configs, resolved by `src/utils.get_device` |
| Seed / determinism | 42, with `cudnn.deterministic=True` | `src/utils.set_seed`, called once at app start |
| Default enhancement method | ESRGAN only | matches the reported benchmark numbers (Hybrid blending is still available in the sidebar) |
| Default inference mode | Quick preview (≤512px long side) | matches the resolution used to score the curated demo images below |
| Curated demo images | one BACH (`Benign_b093.tif`), one Brain MRI (`glioma__brisc2025_train_00427_gl_co_t1.png`) slide | selected from a per-image scan as the strongest model-vs-bicubic margin per domain |

## Next Steps

1. **Check [CONFIG_GUIDE.md](CONFIG_GUIDE.md)** for configuration options
