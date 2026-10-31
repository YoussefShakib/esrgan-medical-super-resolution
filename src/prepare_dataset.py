\
import argparse
import random
import shutil
from pathlib import Path


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder: Path):
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in VALID_EXTS])


def split_dataset(input_dir: Path, train_output: Path, val_output: Path, val_count: int, seed: int):
    images = list_images(input_dir)
    if len(images) <= val_count:
        raise ValueError("val_count must be smaller than number of images")

    random.seed(seed)
    random.shuffle(images)
    val_images = images[:val_count]
    train_images = images[val_count:]

    train_output.mkdir(parents=True, exist_ok=True)
    val_output.mkdir(parents=True, exist_ok=True)

    for src in train_images:
        shutil.copy2(src, train_output / src.name)
    for src in val_images:
        shutil.copy2(src, val_output / src.name)

    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")


def merge_datasets(inputs, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for input_dir in inputs:
        for img in list_images(Path(input_dir)):
            target = output_dir / f"{Path(input_dir).name}_{img.name}"
            shutil.copy2(img, target)
            copied += 1
    print(f"Copied {copied} images to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--input", required=True)
    split_parser.add_argument("--train-output", required=True)
    split_parser.add_argument("--val-output", required=True)
    split_parser.add_argument("--val-count", type=int, default=100)
    split_parser.add_argument("--seed", type=int, default=42)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--inputs", nargs="+", required=True)
    merge_parser.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.command == "split":
        split_dataset(
            Path(args.input),
            Path(args.train_output),
            Path(args.val_output),
            args.val_count,
            args.seed,
        )
    elif args.command == "merge":
        merge_datasets(args.inputs, Path(args.output))


if __name__ == "__main__":
    main()
