\
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to train_log.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    output_dir = csv_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    if "train_loss" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["train_loss"])
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "train_loss.png", dpi=200)
        plt.close()

    if "g_total" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["g_total"], label="Generator Total")
        plt.plot(df["epoch"], df["d_loss"], label="Discriminator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("ESRGAN Training Losses")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "gan_losses.png", dpi=200)
        plt.close()

    if "val_psnr" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["val_psnr"])
        plt.xlabel("Epoch")
        plt.ylabel("PSNR (dB)")
        plt.title("Validation PSNR")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "val_psnr.png", dpi=200)
        plt.close()

    if "val_ssim" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["val_ssim"])
        plt.xlabel("Epoch")
        plt.ylabel("SSIM")
        plt.title("Validation SSIM")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "val_ssim.png", dpi=200)
        plt.close()

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
