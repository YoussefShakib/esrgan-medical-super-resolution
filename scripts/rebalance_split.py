"""Rebalance the BACH train/val split to stratified 85/15 per class.

The current split (340 train / 60 val) is imbalanced:
  val: 17 Benign / 10 InSitu / 19 Invasive / 14 Normal  (target: 15 each)

This script reads bach_flat/, sorts images deterministically, and creates
a fresh 85-train / 15-val split with exactly 15 images per class in val.

Usage
-----
    python scripts/rebalance_split.py [--dry-run]

  --dry-run   Print the new split without touching the filesystem.

The existing datasets/train_HR and datasets/val_HR are renamed with a
`_original` suffix before the new split is written.  Run with --dry-run
first to verify counts.

Outputs
-------
  datasets/train_HR_stratified/   — 85 images per class (340 total)
  datasets/val_HR_stratified/     — 15 images per class (60 total)
  datasets/split_manifest.json    — full list of which images go where
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

BACH_FLAT = Path("datasets/bach_flat")
TRAIN_OUT = Path("datasets/train_HR_stratified")
VAL_OUT = Path("datasets/val_HR_stratified")
MANIFEST_OUT = Path("datasets/split_manifest.json")

CLASSES = ["Benign", "InSitu", "Invasive", "Normal"]
VAL_PER_CLASS = 15  # → 60 val total
TRAIN_PER_CLASS = 85  # → 340 train total


def _get_class(filename: str) -> str:
    for cls in CLASSES:
        if filename.startswith(cls):
            return cls
    return "Unknown"


def build_split(flat_dir: Path) -> Dict[str, Dict[str, List[Path]]]:
    """Return {'train': {cls: [paths]}, 'val': {cls: [paths]}}."""
    by_class: Dict[str, List[Path]] = {c: [] for c in CLASSES}
    for p in sorted(flat_dir.iterdir()):
        if p.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
            continue
        cls = _get_class(p.name)
        if cls in by_class:
            by_class[cls].append(p)

    split: Dict[str, Dict[str, List[Path]]] = {"train": {}, "val": {}}
    for cls, paths in by_class.items():
        if len(paths) < VAL_PER_CLASS + TRAIN_PER_CLASS:
            raise ValueError(
                f"Class '{cls}' has only {len(paths)} images; "
                f"need at least {VAL_PER_CLASS + TRAIN_PER_CLASS}."
            )
        # Deterministic: last VAL_PER_CLASS images → val, rest → train
        # (sorted alphabetically so the assignment is reproducible)
        split["val"][cls] = paths[-VAL_PER_CLASS:]
        split["train"][cls] = paths[:-VAL_PER_CLASS][:TRAIN_PER_CLASS]
    return split


def write_split(split: Dict[str, Dict[str, List[Path]]], dry_run: bool) -> None:
    manifest: Dict[str, List[str]] = {"train": [], "val": []}

    for subset, out_dir in [("train", TRAIN_OUT), ("val", VAL_OUT)]:
        print(f"\n{subset.upper()} split -> {out_dir}")
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
        for cls in CLASSES:
            paths = split[subset].get(cls, [])
            print(f"  {cls}: {len(paths)} images")
            for src in paths:
                dst = out_dir / src.name
                manifest[subset].append(str(src.name))
                if not dry_run:
                    shutil.copy2(src, dst)

    total_train = sum(len(v) for v in split["train"].values())
    total_val = sum(len(v) for v in split["val"].values())
    print(f"\nTotal: {total_train} train / {total_val} val")

    if not dry_run:
        MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_OUT.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"Manifest written to {MANIFEST_OUT}")


def main():
    parser = argparse.ArgumentParser(description="Rebalance BACH train/val split")
    parser.add_argument("--dry-run", action="store_true", help="Print counts only")
    args = parser.parse_args()

    if not BACH_FLAT.exists():
        raise FileNotFoundError(
            f"bach_flat/ not found at {BACH_FLAT.resolve()}. "
            "Run from the project root."
        )

    print("Scanning bach_flat/ …")
    split = build_split(BACH_FLAT)

    # Print current split for comparison
    print("\nCurrent val_HR class distribution (before rebalancing):")
    val_hr = Path("datasets/val_HR")
    if val_hr.exists():
        for cls in CLASSES:
            n = len(list(val_hr.glob(f"{cls}*")))
            print(f"  {cls}: {n}")

    print("\nProposed new split (stratified 85/15 per class):")
    write_split(split, dry_run=args.dry_run)

    if args.dry_run:
        print("\nDry-run complete. No files were written.")
        print("Re-run without --dry-run to create datasets/train_HR_stratified/ "
              "and datasets/val_HR_stratified/.")
    else:
        print("\nStratified split written.")
        print("To use it, update your YAML configs:")
        print("  train_hr_dir: datasets/train_HR_stratified")
        print("  val_hr_dir: datasets/val_HR_stratified")
        print("\nExisting train_HR / val_HR are unchanged; "
              "new runs should reference the *_stratified directories.")


if __name__ == "__main__":
    main()
