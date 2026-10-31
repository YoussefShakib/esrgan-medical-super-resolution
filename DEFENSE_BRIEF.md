# Thesis Defense — Presentation Brief

> **Hand this entire file to your design/content tool.** It contains the full
> narrative, every verified number, the slide-by-slide structure, talking
> points, and exactly which figure belongs on each slide. Nothing here needs
> the thesis document — it is self-contained.

---

## 0. Project identity

- **Title:** ESRGAN-Based Medical Image Super-Resolution (with a downstream tumour-classification extension)
- **Author:** Youssef Shakib
- **Institution:** German University in Cairo (GUC)
- **Degree:** BSc thesis
- **Defense date:** June 2026
- **One-line summary:** A reproducible benchmark of ×4 super-resolution methods (Bicubic, SRResNet, EDSR, ESRGAN) on medical images (breast histopathology + brain MRI), an honest analysis of when deep SR helps vs. hurts, a GAN-stabilisation fix, a degradation ablation, a research-grade tumour classifier, a controlled study on whether SR improves classification, and an interactive Streamlit review tool.

### Framing rule (must appear and must be respected on every slide)
This is a **research prototype**, not a diagnostic tool. Never say "diagnosis / detected / confirmed / cancer found." Always say "**predicted class** with **confidence score**." No clinical claims.

---

## 1. Core narrative (the story the talk tells)

1. **Problem.** Medical images are expensive to acquire at high resolution; degraded/low-resolution scans lose diagnostic detail. Super-resolution (SR) promises to recover that detail computationally.
2. **Question.** Does modern GAN-based SR (ESRGAN) actually recover *useful* detail on medical images, or does it just *hallucinate* plausible-looking texture? And does any of it help a real downstream task (tumour classification)?
3. **Approach.** Build a clean, reproducible benchmark: 4 methods × 2 medical domains × multiple metrics (pixel-fidelity AND perceptual), plus a classifier and a controlled SR→classification study.
4. **Honest findings.**
   - On histopathology (BACH), **simple bicubic stays competitive on pixel metrics (PSNR/SSIM)** — a result the talk confronts head-on instead of hiding.
   - The original GAN training **destabilised the moment adversarial training began**; a 4-part stabilisation fix corrected it.
   - On brain MRI, **ESRGAN clearly beats bicubic** (+1.23 dB PSNR, +0.020 SSIM) — domain matters.
   - The classifier reaches **99.5%** on brain MRI tumour type but only **~77%** on 4-class histopathology — task difficulty matters.
5. **Contribution.** A rigorous, reproducible, honestly-reported study + a working assistive review app — not an oversold "AI diagnoses cancer" claim.

---

## 2. All verified numbers (use these exactly — do not invent)

### 2.1 SR quality on BACH histopathology (×4, 60 validation images)

**Stratified / class-balanced split (15 images per class) — use this as the headline table:**

| Method   | PSNR (dB) | SSIM   |
|----------|-----------|--------|
| Bicubic  | **28.74** | **0.8212** |
| SRResNet | 27.99     | 0.8079 |
| EDSR     | 27.51     | 0.7813 |
| ESRGAN   | 28.31     | 0.8138 |

**Original / unbalanced split (17 Benign : 10 InSitu : 19 Invasive : 14 Normal):**

| Method   | PSNR (dB) | SSIM   |
|----------|-----------|--------|
| Bicubic  | 29.76     | 0.8157 |
| SRResNet | 29.18     | 0.8073 |
| EDSR     | 28.79     | 0.7855 |
| ESRGAN   | 29.44     | 0.8152 |

*Key point: the ranking is unchanged across both splits → the "bicubic is competitive" result is NOT a class-imbalance artefact.*

### 2.2 GAN stabilisation (original vs Stable-GAN ESRGAN, BACH ×4)

- Original ESRGAN: best val PSNR **29.44 dB at epoch 15** — which is the *exact* epoch adversarial training started. Discriminator loss spiked then saturated within ~3 epochs; generator quality collapsed by epoch 21 (PSNR fell to ~27.15). Root cause: `lr_g = lr_d = 1e-4` let the discriminator out-learn the generator.
- The fix (Stable-GAN): **`lr_d = lr_g / 5`**, a **2-epoch discriminator-only warmup**, an **R1 gradient penalty (weight 10)**, and **gradient clipping (1.0)**.
- Result: stable discriminator loss throughout the GAN phase; best val **PSNR 29.58 dB / SSIM 0.819 at epoch 19** with no collapse.

### 2.3 Degradation ablation (Experiment 2 — ESRGAN, 60 val images, ×4)

| Training degradation | PSNR (dB) | SSIM   | LPIPS (↓ better) |
|----------------------|-----------|--------|------------------|
| Bicubic-only         | 30.19     | 0.8279 | 0.183 |
| Mixed (blur+noise+JPEG+resize) | 29.58 | 0.8186 | 0.206 |

*Interpretation: on a clean bicubic test set, the bicubic-only model wins — but the mixed-degradation model is trained for robustness to realistic corruption, trading clean-test fidelity for generalisation. The trade-off IS the finding.*

### 2.4 Brain MRI super-resolution (ESRGAN vs bicubic, 180 images, ×4)

| Method  | PSNR (dB) | SSIM   |
|---------|-----------|--------|
| Bicubic | 32.37     | 0.8874 |
| ESRGAN  | **33.60** | **0.9076** |
| **Δ (ESRGAN − bicubic)** | **+1.23** | **+0.020** |

*Headline contrast: on MRI, ESRGAN BEATS bicubic on both metrics — the opposite of histopathology. The cleaner, higher-contrast structure of MRI suits the GAN; dense histopathology texture does not.*

### 2.5 Tumour classification (research-grade prediction module)

| Dataset | Architecture | Val accuracy | Top-3 acc | Best epoch |
|---------|--------------|--------------|-----------|------------|
| BACH (4-class histopathology, 60 val) | ResNet50 | **76.7%** | 95.0% | 21 |
| BACH | EfficientNet-B0 | 70.0% | 88.3% | 6 |
| BRISC2025 (4-class brain MRI, 1000 test) | ResNet50 | 96.4% | 99.9% | 26 |
| BRISC2025 | EfficientNet-B0 | **99.5%** | 100% | 36 |

*Key point: 4-class histopathology is genuinely hard (~77%); brain-MRI tumour type is near-solved (99.5%). Honest about both.*

### 2.6 SR → classification controlled study (run this before the defense — see §6)

Research question: *Does ESRGAN super-resolution improve, harm, or not affect downstream classification accuracy?* The same trained classifier is applied to each validation image at three quality levels — **HR (ground truth)**, **LR-bicubic (HR→÷4→×4 bicubic)**, and **ESRGAN SR (HR→÷4→ESRGAN ×4)** — with no re-training. Report top-1 accuracy and macro-F1 per level and state the result honestly (a neutral or negative result is still a valid answer to the question). *Numbers pending — generated by `scripts/sr_classification_study.py`.*

---

## 3. Datasets

- **BACH (Breast Cancer Histology):** 400 high-resolution H&E images (2048×1536 px), 4 tissue classes — **Normal, Benign, In-Situ carcinoma, Invasive carcinoma**, 100 images per class. Stratified split: 85 train / 15 val per class (340 / 60 total). RGB; dense, repetitive texture.
- **BRISC2025 (Brain MRI tumour classification):** 6,000 PNG MRI scans, 4 classes — **glioma (1,401), meningioma (1,635), no-tumour (1,207), pituitary adenoma (1,757)**. Mildly imbalanced (no-tumour is the 20.1% minority → class-weighted loss). Split: 5,000 train / 1,000 test. Grayscale-like, high-contrast, cleaner structure.
- **Degradation pipeline (how LR inputs are made):** HR → bicubic ÷4 → model ×4 for fair, matched comparison. Training optionally uses a *mixed* degradation (random blur, Gaussian noise, JPEG artefacts, mixed resize kernels) to mimic real-world corruption.

---

## 4. Methods & metrics (background slides)

### 4.1 SR model lineage (one slide, left→right evolution)
- **Bicubic** — classical interpolation baseline (no learning).
- **SRResNet** — deep residual CNN, pixel (MSE/Charbonnier) loss.
- **EDSR** — enhanced deep residual network, batch-norm removed, larger capacity.
- **ESRGAN** — RRDB generator + relativistic discriminator + perceptual (VGG) loss + adversarial loss → sharp, realistic texture at the cost of pixel fidelity.

### 4.2 ESRGAN training config (the actual settings used)
Scale ×4 · patch size 192 · eval crop 512 · batch 4 · 100 epochs · `lr_g = lr_d = 1e-4` · adversarial training starts at epoch 15 · loss weights: pixel 1.0, perceptual (VGG) 0.01, adversarial 0.005 · mixed degradation (blur p=0.25, noise p=0.25, JPEG p=0.25). Stable-GAN variant overrides: `lr_d = lr_g/5`, 2-epoch D warmup, R1 penalty 10, grad clip 1.0.

### 4.3 Metrics (define all three)
- **PSNR** (dB, ↑) — pixel-fidelity; rewards smooth, low-error reconstruction.
- **SSIM** (0–1, ↑) — structural similarity; perceived structure.
- **LPIPS** (≈0–1, ↓) — learned perceptual distance in AlexNet feature space; correlates with human judgement of realism. *This is the metric that rewards ESRGAN's sharper texture and is the answer to "but bicubic wins on PSNR."*

### 4.4 Classifier protocol
Transfer learning from ImageNet. **ResNet50** — unfreeze last residual stage (`layer4`) + head (~15M trainable params). **EfficientNet-B0** — unfreeze last MBConv stage + head (~1.15M trainable). AdamW (lr 3e-4, weight decay 1e-4), 5-epoch linear warmup → cosine decay, cross-entropy with label smoothing 0.1, class-weighted loss for BRISC. Early stopping (patience 10) on val accuracy; best checkpoint by top-1. Runtime data-leakage guard (train/val filename stems must be disjoint). Domain-specific augmentation (histology: flips/rotation/colour-jitter; MRI: flips/rotation/affine, no colour-jitter — MRI intensity is meaningful).

---

## 5. Slide-by-slide plan (≈18–20 slides, ~15–18 min)

| # | Slide title | Content / talking points | Figure(s) |
|---|-------------|--------------------------|-----------|
| 1 | **Title** | Title, name, GUC, supervisor, date. Research-prototype disclaimer footer. | — |
| 2 | **Motivation** | Why resolution matters in medical imaging; cost of HR acquisition; the promise & risk of computational SR (hallucination). | (icon/stock) |
| 3 | **Research questions** | (1) Does deep SR recover *useful* detail on medical images? (2) Does it differ by domain? (3) Does SR help a downstream task? | — |
| 4 | **Contributions** | Reproducible 4-method benchmark · LPIPS perceptual eval · GAN-stabilisation fix · degradation ablation · MRI extension · research-grade classifier · SR→classification study · Streamlit review app. | — |
| 5 | **Background: SR lineage** | Bicubic → SRResNet → EDSR → ESRGAN; pixel vs perceptual loss. | diagram (make) |
| 6 | **Datasets** | BACH (400, 4 classes, histopathology) + BRISC2025 (6000, 4 classes, MRI). Show example tiles per class. | sample tiles (make from datasets) |
| 7 | **Method & degradation pipeline** | HR→÷4→model×4; mixed degradation for realism; metrics PSNR/SSIM/LPIPS. | pipeline diagram (make) |
| 8 | **Experiment 1: BACH SR results** | The stratified table (§2.1). The honest headline: **bicubic stays competitive on PSNR/SSIM** — explain *why* (pixel metrics reward smoothness; GAN trades fidelity for sharpness → see LPIPS). | qualitative comparison panel |
| 9 | **Ruling out confounders** | Same ranking on stratified vs original split → not a class-imbalance artefact. | (table or curve) |
| 10 | **GAN instability & the fix** | Diagnose: collapse at epoch 15 (GAN start). Fix: lr_d/5 + D-warmup + R1 + grad-clip. Result: stable, PSNR 29.58 / SSIM 0.819. | training-curve comparison |
| 11 | **Experiment 2: degradation ablation** | Bicubic-only vs mixed; the robustness-vs-clean-fidelity trade-off. | ablation qualitative figure |
| 12 | **Brain MRI extension** | The reversal: **ESRGAN beats bicubic (+1.23 dB / +0.020 SSIM)**. Domain matters — clean MRI structure suits the GAN. | MRI bar chart (make from json) |
| 13 | **Classification: setup** | Research-grade prediction module; ResNet50 & EfficientNet-B0; transfer-learning protocol; disclaimer. | architecture diagram (make) |
| 14 | **Classification: results** | BACH ~77% (hard, 4-class) vs BRISC 99.5% (near-solved). Confusion matrices + ROC. | confusion matrices, ROC (GENERATE) |
| 15 | **Where the model looks (Grad-CAM)** | Show 2–3 correct (attends to nuclei / tumour boundary) + 1 failure (attends to background) as an honesty slide. | Grad-CAM panels (GENERATE) |
| 16 | **SR → classification study** | The research question + the 3-level result; honest interpretation. | sr_study_comparison (GENERATE) |
| 17 | **Streamlit application (demo)** | Domain + method selection, tiled inference, PSNR/SSIM/LPIPS panel, SR change map, Research-Prediction tab, per-case export. Live demo or screenshots. | app before/after + zoom panel |
| 18 | **Limitations** | Bicubic competitive on histopathology; GAN hallucination risk; small BACH set (340 train); possible patient-level leakage in public data; NOT clinically validated. | — |
| 19 | **Future work** | Task-aware SR (classification signal in generator loss); larger/clinical datasets; ×8 SR; real (not synthetic) LR. | — |
| 20 | **Conclusion + Q&A** | Restate contributions and the honest core message. Thank you. | — |
| (B) | **Backup slides** | Full config table, per-class metrics, all training curves, original-split table, EfficientNet vs ResNet, calibration/ECE plots. | extras |

---

## 6. ACTION REQUIRED before the defense — generate the missing figures

The classifier and SR-study figures are **not yet on disk**. Run these two commands (CPU is fine) to create everything in slides 14–16:

```powershell
python scripts/eval_classifiers.py --all
python scripts/sr_classification_study.py
```

This generates, under `figures/classifiers/`:
`{bach,brisc}_{resnet50,efficientnet_b0}/confusion_matrix.png`, `roc_curves.png`,
`calibration.png`, `gradcam_*.png`, plus `sr_study_comparison.png`, and writes
`runs/classifier_results.json` and `runs/sr_classification_study.json` (fill the
§2.6 numbers from the latter).

---

## 7. Anticipated examiner questions (prep these)

- *"Why does bicubic beat your deep models?"* → PSNR/SSIM are pixel-fidelity metrics that reward smooth output; GANs trade fidelity for perceptual sharpness — that's why LPIPS and the MRI result matter. It's an honest, known property of perceptual SR, not a training bug.
- *"Isn't GAN-hallucinated detail dangerous in medicine?"* → Yes — this is exactly why it's framed as a research prototype, why I report LPIPS alongside fidelity, and why the SR→classification study tests whether the texture is actually useful vs. misleading.
- *"Why is histopathology accuracy only 77%?"* → 4-class fine-grained tissue discrimination on 340 training images is genuinely hard even for pathologists; MRI tumour-type is a coarser, higher-contrast task (hence 99.5%).
- *"What's novel?"* → Not a new architecture — a rigorous, reproducible, multi-metric, multi-domain *honest* benchmark + the SR↔task-utility study that most SR papers skip + a stabilisation fix with before/after evidence.
- *"Clinical use?"* → None claimed. Research prototype; not validated against diagnostic standards; vocabulary is deliberately constrained to "predicted class / confidence."

---

## 8. Design direction (for the content/design tool)

- Tone: rigorous, honest, confident. This thesis's strength is intellectual honesty — lean into it, don't hide the bicubic result.
- Each results slide: one big figure + 2–3 bullet takeaways + the single number that matters, bolded.
- Use a consistent colour per method across all charts (e.g. Bicubic = grey, SRResNet = blue, EDSR = green, ESRGAN = red/orange) so the audience tracks them instantly.
- Put the research-prototype disclaimer as a persistent slide footer.
- Keep equations minimal — one PSNR/SSIM/LPIPS definition slide is enough.
- Medical / academic aesthetic: clean, lots of whitespace, sans-serif, restrained accent colour.
