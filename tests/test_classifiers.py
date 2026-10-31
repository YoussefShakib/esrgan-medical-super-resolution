"""Tests for the classifier modules (Phase 1–4).

Covers: model construction, dataset loading, single training step,
leakage assertion, classifier service interface.
Does NOT require trained checkpoints — all tests are self-contained.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestClassifierModels:
    def test_build_resnet50(self):
        from classifiers.models import build_classifier, count_trainable_params
        model = build_classifier("resnet50", num_classes=4)
        assert isinstance(model, torch.nn.Module)
        n = count_trainable_params(model)
        # Should be far fewer than total (most backbone frozen)
        total = sum(p.numel() for p in model.parameters())
        assert n < total, "Trainable params should be a subset of total"
        assert n > 0, "At least the head should be trainable"

    def test_build_efficientnet_b0(self):
        from classifiers.models import build_classifier, count_trainable_params
        model = build_classifier("efficientnet_b0", num_classes=4)
        n = count_trainable_params(model)
        total = sum(p.numel() for p in model.parameters())
        assert n < total
        assert n > 0

    def test_unknown_arch_raises(self):
        from classifiers.models import build_classifier
        with pytest.raises(ValueError, match="Unknown arch"):
            build_classifier("vgg16", num_classes=4)

    def test_resnet50_output_shape(self):
        from classifiers.models import build_classifier
        model = build_classifier("resnet50", num_classes=4)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 4)

    def test_efficientnet_b0_output_shape(self):
        from classifiers.models import build_classifier
        model = build_classifier("efficientnet_b0", num_classes=4)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 4)

    def test_resnet50_5class(self):
        from classifiers.models import build_classifier
        model = build_classifier("resnet50", num_classes=5)
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 5)


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestBACHDataset:
    def test_load_train_split(self):
        from classifiers.dataset import build_bach_dataset, get_transforms, BACH_CLASSES
        ds = build_bach_dataset(_ROOT, "train", get_transforms("bach", "train"))
        assert len(ds) == 340, f"Expected 340, got {len(ds)}"
        assert ds.class_names == BACH_CLASSES

    def test_load_val_split(self):
        from classifiers.dataset import build_bach_dataset, get_transforms
        ds = build_bach_dataset(_ROOT, "val", get_transforms("bach", "val"))
        assert len(ds) == 60, f"Expected 60, got {len(ds)}"

    def test_balanced_classes(self):
        from classifiers.dataset import build_bach_dataset, get_transforms
        from collections import Counter
        ds = build_bach_dataset(_ROOT, "train", get_transforms("bach", "train"))
        counts = Counter(ds.labels)
        assert len(counts) == 4
        assert all(v == 85 for v in counts.values()), f"Unbalanced: {dict(counts)}"

    def test_sample_shape(self):
        from classifiers.dataset import build_bach_dataset, get_transforms
        ds = build_bach_dataset(_ROOT, "val", get_transforms("bach", "val"))
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)
        assert 0 <= label < 4

    def test_invalid_split_raises(self):
        from classifiers.dataset import build_bach_dataset
        with pytest.raises(ValueError, match="split must be"):
            build_bach_dataset(_ROOT, "test")

    def test_no_leakage_between_splits(self):
        from classifiers.dataset import build_bach_dataset
        train_ds = build_bach_dataset(_ROOT, "train")
        val_ds = build_bach_dataset(_ROOT, "val")
        train_stems = {p.stem for p in train_ds.paths}
        val_stems = {p.stem for p in val_ds.paths}
        overlap = train_stems & val_stems
        assert len(overlap) == 0, f"Leakage: {len(overlap)} stems in both splits"


class TestBRISCDataset:
    def test_load_train_split(self):
        from classifiers.dataset import build_brisc_dataset, get_transforms, BRISC_CLASSES
        ds = build_brisc_dataset(_ROOT, "train", get_transforms("brisc", "train"))
        assert len(ds) == 5000, f"Expected 5000, got {len(ds)}"
        assert ds.class_names == BRISC_CLASSES

    def test_load_val_split(self):
        from classifiers.dataset import build_brisc_dataset, get_transforms
        ds = build_brisc_dataset(_ROOT, "val", get_transforms("brisc", "val"))
        assert len(ds) == 1000, f"Expected 1000, got {len(ds)}"

    def test_all_four_classes_present(self):
        from classifiers.dataset import build_brisc_dataset
        from collections import Counter
        ds = build_brisc_dataset(_ROOT, "val")
        counts = Counter(ds.labels)
        assert len(counts) == 4, f"Expected 4 classes, got {len(counts)}"

    def test_sample_shape(self):
        from classifiers.dataset import build_brisc_dataset, get_transforms
        ds = build_brisc_dataset(_ROOT, "val", get_transforms("brisc", "val"))
        img, label = ds[0]
        assert img.shape == (3, 224, 224)
        assert 0 <= label < 4

    def test_no_leakage_between_splits(self):
        from classifiers.dataset import build_brisc_dataset
        train_ds = build_brisc_dataset(_ROOT, "train")
        val_ds = build_brisc_dataset(_ROOT, "val")
        train_stems = {p.stem for p in train_ds.paths}
        val_stems = {p.stem for p in val_ds.paths}
        overlap = train_stems & val_stems
        assert len(overlap) == 0, f"Leakage: {len(overlap)} stems in both splits"


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------

class TestTransforms:
    def test_train_bach_returns_tensor(self):
        from classifiers.dataset import get_transforms
        tf = get_transforms("bach", "train")
        img = Image.new("RGB", (256, 256), color=(128, 64, 32))
        out = tf(img)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 224, 224)

    def test_val_transform_is_deterministic(self):
        from classifiers.dataset import get_transforms
        tf = get_transforms("brisc", "val")
        img = Image.new("RGB", (300, 300), color=(100, 150, 200))
        out1 = tf(img)
        out2 = tf(img)
        assert torch.allclose(out1, out2)

    def test_unknown_domain_raises(self):
        from classifiers.dataset import get_transforms
        with pytest.raises(ValueError, match="Unknown domain"):
            get_transforms("ct_scan", "train")


# ---------------------------------------------------------------------------
# Training step test (single batch, no real training)
# ---------------------------------------------------------------------------

class TestTrainingStep:
    def _make_batch(self, n: int = 4, num_classes: int = 4):
        images = torch.randn(n, 3, 224, 224)
        labels = torch.randint(0, num_classes, (n,))
        return images, labels

    def test_resnet50_single_step(self):
        from classifiers.models import build_classifier
        model = build_classifier("resnet50", num_classes=4)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4
        )
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        images, labels = self._make_batch()
        model.train()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        assert loss.item() > 0

    def test_efficientnet_b0_single_step(self):
        from classifiers.models import build_classifier
        model = build_classifier("efficientnet_b0", num_classes=4)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4
        )
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        images, labels = self._make_batch()
        model.train()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_save_and_load(self, tmp_path):
        from classifiers.models import build_classifier
        from classifiers.dataset import BACH_CLASSES
        model = build_classifier("resnet50", num_classes=4)
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        ckpt_path = ckpt_dir / "best.pth"
        torch.save(
            {
                "epoch": 5,
                "arch": "resnet50",
                "dataset": "bach",
                "num_classes": 4,
                "class_names": BACH_CLASSES,
                "model_state": model.state_dict(),
                "val_acc": 0.75,
                "val_loss": 0.5,
            },
            ckpt_path,
        )
        assert ckpt_path.exists()
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        model2 = build_classifier("resnet50", num_classes=4)
        model2.load_state_dict(ckpt["model_state"])
        assert ckpt["val_acc"] == 0.75
        assert ckpt["class_names"] == BACH_CLASSES


# ---------------------------------------------------------------------------
# Classifier service (no checkpoint required — tests graceful None return)
# ---------------------------------------------------------------------------

class TestClassifierService:
    def test_load_missing_returns_none(self):
        from app_services.classifier_service import load_classifier_service
        # 'nonexistent' dataset has no checkpoint — should return None gracefully
        svc = load_classifier_service("nonexistent_dataset_xyz")
        assert svc is None

    def test_prediction_result_fields(self):
        from app_services.classifier_service import PredictionResult
        result = PredictionResult(
            predicted_class="Normal",
            confidence=0.82,
            probabilities=[0.82, 0.10, 0.05, 0.03],
            class_names=["Normal", "Benign", "InSitu", "Invasive"],
        )
        assert result.predicted_class == "Normal"
        assert abs(result.confidence - 0.82) < 1e-6
        assert len(result.probabilities) == 4

    def test_service_predict_with_synthetic_checkpoint(self, tmp_path):
        """Build a real ClassifierService from a temp checkpoint and run predict."""
        from classifiers.models import build_classifier
        from classifiers.dataset import BACH_CLASSES
        from app_services.classifier_service import ClassifierService

        model = build_classifier("resnet50", num_classes=4)
        device = torch.device("cpu")
        svc = ClassifierService(
            model=model,
            class_names=BACH_CLASSES,
            arch="resnet50",
            dataset_name="bach",
            checkpoint_epoch=1,
            val_acc=0.5,
            device=device,
        )
        img = Image.new("RGB", (256, 256), color=(180, 130, 90))
        result = svc.predict(img)
        assert result.predicted_class in BACH_CLASSES
        assert 0.0 <= result.confidence <= 1.0
        assert abs(sum(result.probabilities) - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Leakage check utility
# ---------------------------------------------------------------------------

class TestLeakageCheck:
    def test_clean_sets_pass(self, tmp_path):
        from classifiers.train import _check_leakage
        train_paths = [tmp_path / f"train_{i}.png" for i in range(5)]
        val_paths = [tmp_path / f"val_{i}.png" for i in range(5)]
        _check_leakage(train_paths, val_paths, "test_dataset")  # should not raise

    def test_overlapping_sets_raise(self, tmp_path):
        from classifiers.train import _check_leakage
        shared = [tmp_path / "img_001.png"]
        train_paths = shared + [tmp_path / f"train_{i}.png" for i in range(3)]
        val_paths = shared + [tmp_path / f"val_{i}.png" for i in range(3)]
        with pytest.raises(RuntimeError, match="Data leakage"):
            _check_leakage(train_paths, val_paths, "test_dataset")
