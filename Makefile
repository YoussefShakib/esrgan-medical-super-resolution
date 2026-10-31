## ============================================================
## Thesis Super-Resolution Project — Makefile
## ============================================================
##
## Prerequisites: .venv/Scripts/python.exe (Windows) or
##               python / python3 on PATH after `source .venv/bin/activate`
##
## All targets run from the project root.  GPU strongly recommended
## for training targets; CPU runtimes are noted per target.
##
## Quick start (evaluation only, no training):
##   make eval-all
##
## Full reproduction (including training):
##   make stable-gan-train
##   make ablation-train   (requires approval — ~12-24h CPU)
##   make mri-train        (requires approval — ~12-30h CPU per model)
##   make eval-all
##   make figures

PYTHON := .venv/Scripts/python.exe

# ── Tests ──────────────────────────────────────────────────────────────
.PHONY: test
test:
	$(PYTHON) -m pytest tests/test_sr_project.py tests/test_classifiers.py -v

.PHONY: test-sr
test-sr:
	$(PYTHON) -m pytest tests/test_sr_project.py -v

.PHONY: test-classifiers
test-classifiers:
	$(PYTHON) -m pytest tests/test_classifiers.py -v

.PHONY: test-metrics
test-metrics:
	$(PYTHON) -m pytest tests/test_sr_project.py -v -k "TestLPIPS or TestMetrics"

.PHONY: test-config
test-config:
	$(PYTHON) -m pytest tests/test_sr_project.py -v -k "TestConfig or TestStableGAN or TestAblation"

.PHONY: test-split
test-split:
	$(PYTHON) -m pytest tests/test_sr_project.py -v -k "TestStratifiedSplit"

# ── Split ──────────────────────────────────────────────────────────────
.PHONY: split-dry-run
split-dry-run:
	$(PYTHON) scripts/rebalance_split.py --dry-run

.PHONY: split-create
split-create:
	$(PYTHON) scripts/rebalance_split.py

# ── Evaluation (fast, no training needed) ──────────────────────────────
.PHONY: eval-bicubic
eval-bicubic:
	$(PYTHON) scripts/eval_bicubic_baseline.py
	@echo "Results: runs/bicubic_baseline/evaluation_results.json"

.PHONY: eval-bicubic-stratified
eval-bicubic-stratified:
	$(PYTHON) scripts/eval_bicubic_baseline.py \
		--config configs/eval_stratified.yaml \
		--output-dir runs/bicubic_stratified

.PHONY: eval-lpips
eval-lpips:
	$(PYTHON) scripts/eval_bicubic_baseline.py \
		--compute-lpips \
		--output-dir runs/bicubic_baseline_lpips
	$(PYTHON) src/evaluate.py \
		--config configs/esrgan_bach_medical.yaml \
		--checkpoint runs/esrgan_bach_x4/checkpoints/best_generator.pth \
		--compute-lpips \
		--output runs/esrgan_eval_lpips.json
	$(PYTHON) src/evaluate.py \
		--config configs/edsr_bach_medical.yaml \
		--checkpoint runs/edsr_bach_x4/checkpoints/best.pth \
		--compute-lpips \
		--output runs/edsr_eval_lpips.json
	$(PYTHON) src/evaluate.py \
		--config configs/srresnet_bach_medical.yaml \
		--checkpoint runs/srresnet_bach_x4/checkpoints/best.pth \
		--compute-lpips \
		--output runs/srresnet_eval_lpips.json

.PHONY: eval-all
eval-all: eval-bicubic
	$(PYTHON) src/evaluate.py \
		--config configs/esrgan_bach_medical.yaml \
		--checkpoint runs/esrgan_bach_x4/checkpoints/best_generator.pth \
		--output runs/esrgan_bach_x4/checkpoints/evaluation_results.json
	$(PYTHON) src/evaluate.py \
		--config configs/edsr_bach_medical.yaml \
		--checkpoint runs/edsr_bach_x4/checkpoints/best.pth \
		--output runs/edsr_bach_x4/checkpoints/evaluation_results.json
	$(PYTHON) src/evaluate.py \
		--config configs/srresnet_bach_medical.yaml \
		--checkpoint runs/srresnet_bach_x4/checkpoints/best.pth \
		--output runs/srresnet_bach_x4/checkpoints/evaluation_results.json
	@echo "All BACH evaluation results updated."

# ── Training (long-running — review runtimes before starting) ──────────
# Estimated CPU runtime: 6-12 hours
.PHONY: stable-gan-train
stable-gan-train:
	$(PYTHON) src/train.py --config configs/esrgan_stable_bach.yaml
	@echo "Stable-GAN run saved to runs/esrgan_stable_v1/"

# Estimated CPU runtime: ~12-24 hours (2 × 100 epochs)
.PHONY: ablation-train
ablation-train:
	$(PYTHON) scripts/ablation_degradation.py --compute-lpips

# Skip training, generate results table + figure from existing checkpoints
.PHONY: ablation-results
ablation-results:
	$(PYTHON) scripts/ablation_degradation.py --skip-training --compute-lpips

# Estimated CPU runtime: ~12-30 hours (15 epochs × 2 models × 5100 images)
.PHONY: mri-train
mri-train:
	$(PYTHON) scripts/train_mri_baselines.py --epochs 15 --compute-lpips

.PHONY: mri-eval
mri-eval:
	$(PYTHON) scripts/train_mri_baselines.py --eval-only --compute-lpips
	@echo "MRI evaluation: runs/mri_evaluation.json"

# ── Figures ────────────────────────────────────────────────────────────
.PHONY: figures
figures: ablation-results mri-eval
	@echo "Thesis figures regenerated in runs/ablation_results/ and runs/"

# ── Classifier training (long-running — see runtimes below) ────────────
# Estimated GPU runtime: 2-6h for all 4 models; CPU: 6-20h
# Run with --all to train resnet50 + efficientnet_b0 on both datasets.
.PHONY: train-classifiers
train-classifiers:
	$(PYTHON) scripts/train_classifiers.py --all --epochs 50 --patience 10
	@echo "Checkpoints saved to runs/classifier_{dataset}_{arch}/"

.PHONY: train-classifier-bach
train-classifier-bach:
	$(PYTHON) scripts/train_classifiers.py --dataset bach --arch resnet50
	$(PYTHON) scripts/train_classifiers.py --dataset bach --arch efficientnet_b0

.PHONY: train-classifier-brisc
train-classifier-brisc:
	$(PYTHON) scripts/train_classifiers.py --dataset brisc --arch resnet50
	$(PYTHON) scripts/train_classifiers.py --dataset brisc --arch efficientnet_b0

# ── Classifier evaluation ───────────────────────────────────────────────
.PHONY: eval-classifiers
eval-classifiers:
	$(PYTHON) scripts/eval_classifiers.py --all
	@echo "Results: runs/classifier_results.json"
	@echo "Figures: figures/classifiers/"

# ── SR–classification study ─────────────────────────────────────────────
.PHONY: sr-study
sr-study:
	$(PYTHON) scripts/sr_classification_study.py
	@echo "Results: runs/sr_classification_study.json"
	@echo "Plot:    figures/classifiers/sr_study_comparison.png"

# ── Help ───────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  test                  Run full pytest suite (SR + classifier)"
	@echo "  test-sr               Run SR pipeline tests only"
	@echo "  test-classifiers      Run classifier tests only"
	@echo "  test-metrics          Run LPIPS / PSNR / SSIM tests only"
	@echo "  test-config           Run config / stable-GAN / ablation tests"
	@echo "  test-split            Run stratified split tests"
	@echo "  split-dry-run         Preview stratified split (no file changes)"
	@echo "  split-create          Create datasets/train_HR_stratified/ + val_HR_stratified/"
	@echo "  eval-bicubic          Bicubic baseline evaluation on original val set"
	@echo "  eval-bicubic-stratified  Bicubic baseline on stratified val set"
	@echo "  eval-lpips            Evaluate all 4 models with LPIPS on original val"
	@echo "  eval-all              Re-run evaluation_results.json for all 4 models"
	@echo "  stable-gan-train      Train ESRGAN with stable-GAN schedule (CPU: 6-12h)"
	@echo "  ablation-train        Run degradation ablation — 2 ESRGAN runs (CPU: 12-24h)"
	@echo "  ablation-results      Regenerate ablation table + figure from checkpoints"
	@echo "  mri-train             Train EDSR + SRResNet on Brain MRI (CPU: 12-30h)"
	@echo "  mri-eval              Evaluate all 4 methods on Brain MRI (checkpoints needed)"
	@echo "  figures               Regenerate all thesis figures from existing checkpoints"
	@echo "  train-classifiers     Train all 4 classifiers (GPU: 2-6h, CPU: 6-20h)"
	@echo "  train-classifier-bach Train BACH classifiers only"
	@echo "  train-classifier-brisc Train BRISC classifiers only"
	@echo "  eval-classifiers      Evaluate all trained classifiers + Grad-CAM"
	@echo "  sr-study              SR–classification controlled study (Phase 3)"
