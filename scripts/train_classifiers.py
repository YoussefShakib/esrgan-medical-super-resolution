"""CLI for training BACH and BRISC2025 classifiers.

Usage examples
--------------
# Train all four combinations (ResNet50 + EffNet-B0 × BACH + BRISC):
    python scripts/train_classifiers.py --all

# Train a single model:
    python scripts/train_classifiers.py --dataset bach --arch resnet50

# Override training hyper-parameters:
    python scripts/train_classifiers.py --dataset brisc --arch efficientnet_b0 \\
        --epochs 60 --batch-size 32 --lr 1e-4 --patience 15

Checkpoints are saved to  runs/classifier_{dataset}_{arch}/checkpoints/best.pth
Training logs are saved to runs/classifier_{dataset}_{arch}/train_log.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from classifiers.train import train_classifier  # noqa: E402

DATASETS = ["bach", "brisc"]
ARCHS = ["resnet50", "efficientnet_b0"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train BACH / BRISC2025 image classifiers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", choices=DATASETS, help="Dataset to train on")
    p.add_argument("--arch", choices=ARCHS, help="Model architecture")
    p.add_argument(
        "--all", action="store_true", help="Train all dataset × arch combinations"
    )
    p.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size")
    p.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    p.add_argument(
        "--patience", type=int, default=10, help="Early-stopping patience"
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: runs/classifier_{dataset}_{arch})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.all:
        combos = [(d, a) for d in DATASETS for a in ARCHS]
    elif args.dataset and args.arch:
        combos = [(args.dataset, args.arch)]
    else:
        print(
            "ERROR: specify --dataset and --arch, or use --all\n"
            f"  datasets: {DATASETS}\n"
            f"  archs:    {ARCHS}"
        )
        sys.exit(1)

    print(f"Training {len(combos)} model(s): {combos}")
    print(
        f"  epochs={args.epochs}  batch_size={args.batch_size}"
        f"  lr={args.lr}  patience={args.patience}  seed={args.seed}"
    )

    results = {}
    for dataset_name, arch in combos:
        print(f"\n{'='*60}")
        print(f"  dataset={dataset_name}  arch={arch}")
        print(f"{'='*60}")
        out_dir = args.output_dir or (
            PROJECT_ROOT / "runs" / f"classifier_{dataset_name}_{arch}"
        )
        metrics = train_classifier(
            dataset_name=dataset_name,
            arch=arch,
            project_root=PROJECT_ROOT,
            output_dir=out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            seed=args.seed,
        )
        results[f"{dataset_name}_{arch}"] = metrics

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for key, m in results.items():
        print(
            f"  {key:40s}  val_acc={m.get('val_acc', '?'):.4f}"
            f"  top3={m.get('val_acc_top3', '?'):.4f}"
            f"  epoch={m.get('epoch', '?')}"
        )


if __name__ == "__main__":
    main()
