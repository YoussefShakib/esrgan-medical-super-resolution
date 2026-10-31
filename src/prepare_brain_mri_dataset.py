"""Prepare a brain-MRI super-resolution dataset.

Supports both of these source layouts:

1. Flat class folders:
   ``raw/glioma/*.jpg``, ``raw/meningioma/*.jpg``, ...
2. Split + class folders:
   ``classification_task/train/glioma/*.jpg``
   ``classification_task/test/glioma/*.jpg``

The script:

- recursively finds image files only
- ignores segmentation / mask / annotation style paths
- converts every image to 3-channel RGB
- optionally stages RGB copies into ``raw/``
- writes flattened PNGs into ``train_HR`` / ``val_HR``
- preserves class names in output filenames to avoid collisions

Example
-------

    python src/prepare_brain_mri_dataset.py \
        --input "C:\\Users\\shaki\\Downloads\\archive\\brisc2025" \
        --raw-output datasets/BRAIN_MRI/raw \
        --train-output datasets/BRAIN_MRI/train_HR \
        --val-output datasets/BRAIN_MRI/val_HR \
        --val-ratio 0.15 \
        --seed 42 \
        --clean
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from PIL import Image


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASSIFICATION_DIR_CANDIDATES = ("classification_task", "classification", "Classification")
SPLIT_DIR_NAMES = {"train", "test", "val", "validation"}
EXCLUDED_KEYWORDS = {
    "segmentation",
    "mask",
    "masks",
    "annotation",
    "annotations",
    "label",
    "labels",
}


@dataclass(frozen=True)
class ImageRecord:
    class_label: str
    split_label: str
    path: Path
    source_root: Path


def _resolve_input_root(input_dir: Path) -> Path:
    """Resolve BRISC-style dataset roots to the classification image folder."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    for candidate in CLASSIFICATION_DIR_CANDIDATES:
        candidate_path = input_dir / candidate
        if candidate_path.is_dir():
            return candidate_path
    return input_dir


def _has_excluded_component(rel_path: Path) -> bool:
    parts = [part.lower() for part in rel_path.parts]
    for part in parts[:-1]:
        if any(keyword in part for keyword in EXCLUDED_KEYWORDS):
            return True
    return False


def _class_and_split_from_relative_path(rel: Path) -> tuple[str, str]:
    """Infer class and split labels from a path below the chosen source root."""
    parts = list(rel.parts)
    if not parts:
        return "unknown", "source"

    if parts[0].lower() in SPLIT_DIR_NAMES and len(parts) >= 2:
        return parts[1], parts[0].lower()

    return parts[0], "source"


def _list_class_images(input_dir: Path) -> tuple[Path, List[ImageRecord]]:
    """Return detected source root and image records under it."""
    source_root = _resolve_input_root(input_dir)
    items: List[ImageRecord] = []

    for path in source_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_EXTS:
            continue

        rel = path.relative_to(source_root)
        if _has_excluded_component(rel):
            continue

        class_label, split_label = _class_and_split_from_relative_path(rel)
        items.append(
            ImageRecord(
                class_label=class_label,
                split_label=split_label,
                path=path,
                source_root=source_root,
            )
        )

    items.sort(key=lambda item: (item.class_label.lower(), item.path.as_posix()))
    return source_root, items


def _safe_target_name(record: ImageRecord, taken: set[str]) -> str:
    """Return a collision-free filename with class name and source context."""
    rel = record.path.relative_to(record.source_root)
    parts = list(rel.parts[:-1])

    if parts and parts[0].lower() in SPLIT_DIR_NAMES:
        parts = parts[1:]
    if parts and parts[0].lower() == record.class_label.lower():
        parts = parts[1:]

    safe_parts = [record.class_label]
    safe_parts.extend(part.replace(" ", "_") for part in parts if part)
    safe_parts.append(record.path.stem.replace(" ", "_"))

    base = "__".join(safe_parts)
    candidate = f"{base}.png"
    n = 1
    while candidate in taken:
        candidate = f"{base}_dup{n}.png"
        n += 1
    taken.add(candidate)
    return candidate


def _save_as_rgb_png(src: Path, dst: Path) -> None:
    with Image.open(src) as im:
        rgb = im.convert("RGB")
        dst.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(dst, format="PNG")


def _split_indices(
    n: int,
    val_ratio: float | None,
    val_count: int | None,
    seed: int,
) -> tuple[List[int], List[int]]:
    if val_ratio is None and val_count is None:
        raise ValueError("Either --val-ratio or --val-count must be provided.")
    if val_ratio is not None and val_count is not None:
        raise ValueError("--val-ratio and --val-count are mutually exclusive.")
    if val_count is None:
        val_count = max(1, int(round(n * float(val_ratio))))
    if val_count >= n:
        raise ValueError(f"val_count={val_count} must be smaller than total image count {n}.")

    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    val_idx = sorted(indices[:val_count])
    train_idx = sorted(indices[val_count:])
    return train_idx, val_idx


def _clean_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def prepare(
    input_dir: Path,
    train_output: Path,
    val_output: Path,
    val_ratio: float | None,
    val_count: int | None,
    seed: int,
    clean: bool,
    raw_output: Path | None = None,
) -> None:
    source_root, items = _list_class_images(input_dir)
    if not items:
        raise RuntimeError(
            f"No images found under {source_root} with extensions {sorted(VALID_EXTS)}."
        )

    class_counts = Counter(item.class_label for item in items)
    train_idx, val_idx = _split_indices(len(items), val_ratio, val_count, seed)

    print(f"Detected source folder: {source_root}")
    print(f"Total MRI image files found: {len(items)}")
    print(f"Detected class folders: {', '.join(sorted(class_counts))}")
    for cls, n in sorted(class_counts.items()):
        print(f"  {cls}: {n}")
    print(f"Files planned for train_HR: {len(train_idx)}")
    print(f"Files planned for val_HR: {len(val_idx)}")
    if raw_output is not None:
        print(f"Raw staging folder: {raw_output}")

    if clean:
        for d in (raw_output, train_output, val_output):
            if d is not None:
                _clean_output_dir(d)

    if raw_output is not None:
        raw_output.mkdir(parents=True, exist_ok=True)
    train_output.mkdir(parents=True, exist_ok=True)
    val_output.mkdir(parents=True, exist_ok=True)

    raw_written = 0
    if raw_output is not None:
        for record in items:
            raw_name = f"{record.path.stem.replace(' ', '_')}.png"
            raw_dst = raw_output / record.class_label / record.split_label / raw_name
            _save_as_rgb_png(record.path, raw_dst)
            raw_written += 1

    train_taken: set[str] = set()
    val_taken: set[str] = set()
    train_classes: Counter[str] = Counter()
    val_classes: Counter[str] = Counter()

    for i in train_idx:
        record = items[i]
        name = _safe_target_name(record, train_taken)
        _save_as_rgb_png(record.path, train_output / name)
        train_classes[record.class_label] += 1

    for i in val_idx:
        record = items[i]
        name = _safe_target_name(record, val_taken)
        _save_as_rgb_png(record.path, val_output / name)
        val_classes[record.class_label] += 1

    if raw_output is not None:
        print(f"Raw RGB files written: {raw_written}  ->  {raw_output}")
    print(f"Train images written: {len(train_idx)}  ->  {train_output}")
    for cls, n in sorted(train_classes.items()):
        print(f"  train {cls}: {n}")
    print(f"Val images written: {len(val_idx)}  ->  {val_output}")
    for cls, n in sorted(val_classes.items()):
        print(f"  val {cls}: {n}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare a brain-MRI SR dataset.")
    parser.add_argument(
        "--input",
        required=True,
        help="Brain-MRI dataset root, classification folder, or raw folder.",
    )
    parser.add_argument(
        "--raw-output",
        default=None,
        help="Optional RGB staging folder for copied classification images.",
    )
    parser.add_argument(
        "--train-output",
        required=True,
        help="Output train_HR folder.",
    )
    parser.add_argument(
        "--val-output",
        required=True,
        help="Output val_HR folder.",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--val-ratio", type=float, default=None, help="Validation fraction (e.g. 0.15).")
    group.add_argument("--val-count", type=int, default=None, help="Absolute validation image count.")

    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for reproducibility.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing raw/train/val output directories before writing.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    val_ratio = args.val_ratio
    val_count = args.val_count
    if val_ratio is None and val_count is None:
        val_ratio = 0.15

    prepare(
        input_dir=Path(args.input),
        raw_output=Path(args.raw_output) if args.raw_output else None,
        train_output=Path(args.train_output),
        val_output=Path(args.val_output),
        val_ratio=val_ratio,
        val_count=val_count,
        seed=args.seed,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
