# ESRGAN-Based Medical Image Super-Resolution

**BSc thesis project — German University in Cairo (GUC)**
Author: Youssef Shakib

A reproducible benchmark of ×4 super-resolution (Bicubic, SRResNet, EDSR, ESRGAN) on two
medical domains — breast histopathology (BACH) and brain MRI (BRISC2025) — plus a GAN
stabilisation fix, a degradation ablation, a tumour classifier, a controlled study of
whether super-resolution actually helps downstream classification, and a Streamlit app
for assistive visual review.

> **Research prototype, not a diagnostic tool.** Outputs are *predicted classes with
> confidence scores*, never diagnoses. No clinical claims are made or implied.

---

## Headline findings

**1. On histopathology, plain bicubic stays competitive.** Across both the original and
the class-balanced validation split, bicubic wins on PSNR/SSIM — the ranking is identical
on both splits, so this is not a class-imbalance artefact.

**2. On brain MRI, ESRGAN clearly wins.** +1.23 dB PSNR and +0.020 SSIM over bicubic — the
opposite of histopathology. Clean, high-contrast MRI structure suits a GAN; dense,
repetitive histology texture does not. **Domain matters more than architecture.**

**3. The original GAN training collapsed, and the fix is documented.** Best PSNR landed at
exactly the epoch adversarial training began; the discriminator out-learned the generator
within ~3 epochs. A 4-part stabilisation (`lr_d = lr_g/5`, 2-epoch D warmup, R1 penalty,
gradient clipping) removed the collapse.

**4. Super-resolution's value depends entirely on how degraded the input was.** Applied to
already-clean images it slightly hurts classification; applied to genuinely degraded MRI it
recovers most of the lost accuracy (48.3% → 87.7%).

---

## Results

### SR quality — BACH histopathology (×4, 60 validation images)

Class-balanced split (15 images per class) — headline table:

| Method   | PSNR (dB)  | SSIM       |
|----------|------------|------------|
| Bicubic  | **28.74**  | **0.8212** |
| SRResNet | 27.99      | 0.8079     |
| EDSR     | 27.51      | 0.7813     |
| ESRGAN   | 28.31      | 0.8138     |

Original unbalanced split (17 Benign : 10 InSitu : 19 Invasive : 14 Normal):

| Method   | PSNR (dB) | SSIM   |
|----------|-----------|--------|
| Bicubic  | 29.76     | 0.8157 |
| SRResNet | 29.18     | 0.8073 |
| EDSR     | 28.79     | 0.7855 |
| ESRGAN   | 29.44     | 0.8152 |

### SR quality — Brain MRI (×4, 180 images)

| Method                   | PSNR (dB)  | SSIM       |
|--------------------------|------------|------------|
| Bicubic                  | 32.37      | 0.8874     |
| ESRGAN                   | **33.60**  | **0.9076** |
| **Δ (ESRGAN − bicubic)** | **+1.23**  | **+0.020** |

### GAN stabilisation (BACH ×4)

| Variant            | Best val PSNR | Best SSIM | Outcome                          |
|--------------------|---------------|-----------|----------------------------------|
| Original ESRGAN    | 29.44 dB @ ep 15 | 0.8152 | Collapsed by epoch 21 (~27.15 dB) |
| Stable-GAN ESRGAN  | **29.58 dB @ ep 19** | **0.819** | Stable throughout GAN phase   |

Fix: `lr_d = lr_g / 5` · 2-epoch discriminator-only warmup · R1 gradient penalty (weight 10) · gradient clipping (1.0)

### Degradation ablation (ESRGAN, 60 val images, ×4)

| Training degradation           | PSNR (dB) | SSIM   | LPIPS ↓ |
|--------------------------------|-----------|--------|---------|
| Bicubic-only                   | **30.19** | **0.8279** | **0.183** |
| Mixed (blur+noise+JPEG+resize) | 29.58     | 0.8186 | 0.206   |

On a clean bicubic test set the bicubic-only model wins — but the mixed-degradation model
buys robustness to realistic corruption. The trade-off *is* the finding.

### Tumour classification

| Dataset                               | Architecture    | Val top-1 | Top-3  | Best epoch |
|---------------------------------------|-----------------|-----------|--------|------------|
| BACH (4-class histopathology, 60 val) | ResNet50        | **76.7%** | 95.0%  | 21 |
| BACH                                  | EfficientNet-B0 | 70.0%     | 88.3%  | 6  |
| BRISC2025 (4-class brain MRI, 1000 test) | ResNet50     | 96.4%     | 99.9%  | 26 |
| BRISC2025                             | EfficientNet-B0 | **99.5%** | 100%   | 36 |

4-class histopathology is genuinely hard (~77%); brain-MRI tumour type is near-solved (99.5%).

### SR → classification controlled study

The *same* trained classifier is applied to every validation image at three quality levels,
with no re-training: original HR, degraded LR (HR → ÷4 bicubic → ×4 bicubic), and
ESRGAN SR (HR → ÷4 → ESRGAN ×4).

| Dataset | Condition             | Top-1     | Macro-F1 |
|---------|-----------------------|-----------|----------|
| BACH    | Original HR           | **76.7%** | **0.765** |
| BACH    | Degraded LR (bicubic) | 75.0%     | 0.742    |
| BACH    | ESRGAN SR             | 73.3%     | 0.733    |
| BRISC   | Original HR           | **99.5%** | **0.995** |
| BRISC   | Degraded LR (bicubic) | 48.3%     | 0.459    |
| BRISC   | ESRGAN SR             | 87.7%     | 0.863    |

**Interpretation.** On BACH the three conditions are within a few points — degradation
barely hurts and SR does not help, because 2048×1536 histology patches carry far more
texture than the classifier needs. On BRISC the picture is dramatic: degradation destroys
accuracy (99.5% → 48.3%), and ESRGAN restores most of it (→ 87.7%), recovering roughly
77% of the lost accuracy without any classifier re-training. **Super-resolution is worth
it when the input is genuinely degraded, and counter-productive when it is not** — a
negative result on one domain and a strong positive on the other, reported honestly.

Raw per-image predictions: `runs/sr_classification_study.json`

---

## Repository layout

```
├── configs/                  YAML training configs (ESRGAN / EDSR / SRResNet, both domains)
├── src/
│   ├── core/                 Config, checkpoint, device, logging, exceptions
│   ├── models/               ESRGAN (RRDB), EDSR, SRResNet, SRCNN, Discriminator
│   ├── classifiers/          ResNet50 / EfficientNet-B0 transfer-learning module
│   ├── train.py              Training entry point (reconstruction + GAN phases)
│   ├── evaluate.py           Standalone evaluation
│   ├── metrics.py            PSNR, SSIM, LPIPS (AlexNet)
│   ├── dataset.py            Dataset + degradation pipeline
│   └── losses.py             Charbonnier, VGG feature extractor
├── scripts/
│   ├── eval_bicubic_baseline.py    Fast bicubic eval (no checkpoint needed)
│   ├── rebalance_split.py          Stratified 85/15 BACH split
│   ├── ablation_degradation.py     Degradation ablation
│   ├── train_classifiers.py        Classifier training
│   ├── sr_classification_study.py  SR → classification controlled study
│   └── gen_defense_figures.py      Regenerate thesis figures
├── app.py                    Streamlit review interface
├── app_services/             SR, metrics, change-map, findings, classifier services
├── tests/                    pytest suite (SR project + classifiers)
├── runs/                     Metrics, training curves, logs, plots, used-configs
├── figures/                  Thesis figures
├── thesis_outputs/           Export-ready panels
├── DEFENSE_BRIEF.md          Full thesis narrative + every verified number
├── CONFIG_GUIDE.md           Config field reference
└── INSTALL.md                Detailed setup
```

Note: `datasets/` and `*.pth` checkpoints are **not** in this repository — see the two
sections below.

---

## Obtaining the data

The datasets are **not redistributed here**; both carry their own licence and
redistribution terms. Download them from the original sources:

- **BACH — ICIAR 2018 Grand Challenge on Breast Cancer Histology.**
  400 H&E images at 2048×1536, 4 classes (Normal, Benign, In-Situ, Invasive), 100 each.
  <https://iciar2018-challenge.grand-challenge.org/>
- **BRISC2025 — brain MRI tumour classification.**
  ~6,000 PNG scans, 4 classes (glioma, meningioma, no-tumour, pituitary adenoma).

Expected layout once downloaded:

```
datasets/
├── bach_flat/              All 400 BACH images, flat
├── train_HR/               340-image train split
├── val_HR/                 60-image val split
├── train_HR_stratified/    Balanced split — 85/class
├── val_HR_stratified/      Balanced split — 15/class
└── BRAIN_MRI/              5000 train / 1000 test MRI images
```

Build the balanced BACH split with:

```powershell
python scripts/rebalance_split.py --dry-run   # preview
python scripts/rebalance_split.py             # writes datasets/*_stratified/
```

## Pre-trained weights

Checkpoints are 16–90 MB each (~1.5 GB in total), so they are published as
**[release assets](../../releases/latest)** rather than tracked in git. The release
carries the four weights needed to run the app and reproduce the headline numbers:

| Asset                            | Model                        | Size  |
|----------------------------------|------------------------------|-------|
| `esrgan_bach_best_generator.pth` | ESRGAN ×4, BACH              | 87 MB |
| `esrgan_mri_best_generator.pth`  | ESRGAN ×4, Brain MRI         | 87 MB |
| `classifier_bach_resnet50.pth`   | BACH classifier (76.7%)      | 90 MB |
| `classifier_brisc_efficientnet_b0.pth` | BRISC classifier (99.5%) | 16 MB |

Download them into the layout the configs expect:

```
runs/esrgan_bach_x4/checkpoints/best_generator.pth
runs/esrgan_brain_mri_x4/checkpoints/best_generator.pth
runs/classifier_bach_resnet50/checkpoints/best.pth
runs/classifier_brisc_efficientnet_b0/checkpoints/best.pth
```

---

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Install PyTorch for your CUDA version first: https://pytorch.org
pip install -r requirements.txt
pip install pytest
```

**GPU strongly recommended** for training. All evaluation scripts run fine on CPU.

## Quick start (no GPU, no checkpoints needed)

```powershell
pip install -r requirements.txt
python scripts/eval_bicubic_baseline.py          # ~5 seconds
```

## Training

```powershell
# Original ESRGAN (BACH)
python src/train.py --config configs/esrgan_bach_medical.yaml

# Stable-GAN ESRGAN — the stabilisation fix
python src/train.py --config configs/esrgan_stable_bach.yaml
# or apply the fix to any config:
python src/train.py --config configs/esrgan_bach_medical.yaml --stable-gan

# EDSR / SRResNet baselines (BACH)
python src/train.py --config configs/edsr_bach_medical.yaml
python src/train.py --config configs/srresnet_bach_medical.yaml

# Degradation ablation (trains both configs, then evaluates)
python scripts/ablation_degradation.py --compute-lpips

# Brain MRI baselines
python scripts/train_mri_baselines.py --epochs 15 --compute-lpips

# Classifiers
python scripts/train_classifiers.py
```

## Evaluation

```powershell
# Bicubic baseline (fast)
python scripts/eval_bicubic_baseline.py

# Bicubic on the stratified split
python scripts/eval_bicubic_baseline.py `
  --config configs/eval_stratified.yaml `
  --output-dir runs/bicubic_stratified

# Any trained model
python src/evaluate.py `
  --config configs/esrgan_bach_medical.yaml `
  --checkpoint runs/esrgan_bach_x4/checkpoints/best_generator.pth `
  --compute-lpips

# SR -> classification controlled study
python scripts/sr_classification_study.py
```

## Tests

```powershell
python -m pytest tests/ -v
```

Covers config loading, dataset behaviour, model forward passes, checkpoint save/load,
PSNR/SSIM/LPIPS metrics, stable-GAN config fields, stratified-split balance, ablation
config consistency, and the classifier module (including a train/val leakage guard).

## Makefile targets

```
make test                  run the full pytest suite
make eval-bicubic          bicubic baseline (seconds)
make eval-lpips            all 4 models with LPIPS
make eval-all              regenerate evaluation_results.json for all models
make stable-gan-train      train stable-GAN ESRGAN
make ablation-train        degradation ablation (2 ESRGAN runs)
make ablation-results      regenerate table + figure from existing checkpoints
make mri-train             train EDSR + SRResNet on Brain MRI
make mri-eval              evaluate all 4 methods on MRI
make figures               regenerate all thesis figures
make help                  full target list
```

## Streamlit application

```powershell
streamlit run app.py
```

Domain selection (BACH / Brain MRI / General), method selection (ESRGAN / hybrid
bicubic+ESRGAN / bicubic), tiled inference for large images, PSNR/SSIM/LPIPS panel,
SR change map, research prediction tab, and per-case export with a `report.md`.

## Runtime notes

| Task                           | GPU (RTX 3060) | CPU only |
|--------------------------------|----------------|----------|
| Bicubic eval (60 images)       | ~3 s           | ~5 s     |
| Model eval (60 images, ESRGAN) | ~20 s          | ~2 min   |
| ESRGAN train (100 ep, BACH)    | ~30 min        | ~6–12 h  |
| ESRGAN train (15 ep, MRI)      | ~45 min        | ~8–20 h  |
| EDSR train (15 ep, MRI)        | ~30 min        | ~6–15 h  |

---

## Methods

**Model lineage.** Bicubic (classical interpolation, no learning) → SRResNet (deep residual
CNN, pixel loss) → EDSR (batch-norm removed, larger capacity) → ESRGAN (RRDB generator,
relativistic discriminator, VGG perceptual loss, adversarial loss).

**ESRGAN config as used.** ×4 scale · patch 192 · eval crop 512 · batch 4 · 100 epochs ·
`lr_g = lr_d = 1e-4` · adversarial phase starts at epoch 15 · loss weights pixel 1.0,
perceptual 0.01, adversarial 0.005 · mixed degradation (blur / noise / JPEG, each p=0.25).

**Metrics.** PSNR (dB ↑, pixel fidelity) · SSIM (↑, perceived structure) · LPIPS (↓,
learned perceptual distance in AlexNet feature space — the metric that credits ESRGAN's
sharper texture and the direct answer to "but bicubic wins on PSNR").

**Classifier protocol.** ImageNet transfer learning. ResNet50 unfreezes `layer4` + head
(~15M trainable); EfficientNet-B0 unfreezes the last MBConv stage + head (~1.15M). AdamW
(lr 3e-4, wd 1e-4), 5-epoch linear warmup → cosine decay, cross-entropy with label
smoothing 0.1, class-weighted loss for BRISC. Early stopping (patience 10) on val top-1.
Runtime data-leakage guard asserts train/val filename stems are disjoint. Domain-specific
augmentation: histology gets flips/rotation/colour-jitter, MRI gets flips/rotation/affine
but no colour-jitter, since MRI intensity is meaningful.

---

## Licence

Code released under the [MIT Licence](LICENSE). The BACH and BRISC2025 datasets are **not**
covered by this licence and remain subject to their original terms.
