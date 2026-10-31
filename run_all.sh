#!/usr/bin/env bash
# run_all.sh — Full pipeline reproduction script
#
# RESEARCH PROTOTYPE ONLY — not a clinical tool.
#
# Reproduces the complete thesis pipeline in order:
#   1. SR training (ESRGAN BACH + Brain MRI)
#   2. SR evaluation (PSNR/SSIM/LPIPS on all models)
#   3. Classifier training (ResNet50 + EfficientNet-B0 on BACH + BRISC2025)
#   4. Classifier evaluation (confusion matrix, ROC, Grad-CAM, calibration)
#   5. SR–classification study (HR vs LR vs ESRGAN-SR accuracy comparison)
#
# Usage:
#   bash run_all.sh              # full pipeline
#   bash run_all.sh --skip-sr    # skip SR training (use existing checkpoints)
#   bash run_all.sh --skip-train # skip both SR and classifier training
#
# Runtime estimates (GPU strongly recommended):
#   SR training:           6-30 hours depending on model
#   Classifier training:   2-6 hours (GPU) / 6-20 hours (CPU)
#   SR study:              30-90 minutes (runs SR inference on val set)
#
# Requires:
#   .venv/Scripts/python.exe (Windows venv) or python on PATH
#   pip install -r requirements.txt

set -e

PYTHON=".venv/Scripts/python.exe"
if ! [ -f "$PYTHON" ]; then
    PYTHON="python"
fi

SKIP_SR=false
SKIP_TRAIN=false

for arg in "$@"; do
    case $arg in
        --skip-sr)    SKIP_SR=true ;;
        --skip-train) SKIP_SR=true; SKIP_TRAIN=true ;;
    esac
done

echo "================================================================"
echo "  esr_med — Full Pipeline Reproduction"
echo "  RESEARCH PROTOTYPE ONLY — not a clinical tool"
echo "================================================================"
echo ""

# ── Step 0: Tests ──────────────────────────────────────────────────────
echo "[0/5] Running test suite …"
$PYTHON -m pytest tests/test_sr_project.py tests/test_classifiers.py -q
echo ""

# ── Step 1: SR Training ────────────────────────────────────────────────
if [ "$SKIP_SR" = false ]; then
    echo "[1/5] SR training — BACH ESRGAN (stable-GAN schedule) …"
    $PYTHON src/train.py --config configs/esrgan_stable_bach.yaml
    echo ""
    echo "[1/5] SR training — Brain MRI ESRGAN …"
    $PYTHON src/train.py --config configs/esrgan_brain_mri.yaml
    echo ""
else
    echo "[1/5] SR training SKIPPED (--skip-sr)"
fi

# ── Step 2: SR Evaluation ──────────────────────────────────────────────
echo "[2/5] SR evaluation (PSNR/SSIM on all models) …"
$PYTHON scripts/eval_bicubic_baseline.py \
    --config configs/eval_stratified.yaml \
    --output-dir runs/bicubic_stratified
$PYTHON src/evaluate.py \
    --config configs/esrgan_bach_medical.yaml \
    --checkpoint runs/esrgan_bach_x4/checkpoints/best_generator.pth \
    --output runs/esrgan_bach_x4/checkpoints/evaluation_results.json
echo ""

# ── Step 3: Classifier Training ────────────────────────────────────────
if [ "$SKIP_TRAIN" = false ]; then
    echo "[3/5] Classifier training — all 4 models (this may take several hours) …"
    $PYTHON scripts/train_classifiers.py --all --epochs 50 --patience 10
    echo ""
else
    echo "[3/5] Classifier training SKIPPED (--skip-train)"
fi

# ── Step 4: Classifier Evaluation ─────────────────────────────────────
echo "[4/5] Classifier evaluation (confusion matrix, ROC, Grad-CAM) …"
$PYTHON scripts/eval_classifiers.py --all
echo ""

# ── Step 5: SR–Classification Study ───────────────────────────────────
echo "[5/5] SR–classification study (HR vs LR vs ESRGAN-SR) …"
$PYTHON scripts/sr_classification_study.py
echo ""

echo "================================================================"
echo "  Pipeline complete."
echo ""
echo "  SR results:          runs/bach_evaluation_summary.json"
echo "  Classifier results:  runs/classifier_results.json"
echo "  SR study results:    runs/sr_classification_study.json"
echo "  Figures:             figures/classifiers/"
echo "================================================================"
