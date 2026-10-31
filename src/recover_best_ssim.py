"""Retroactively recover a best-SSIM-post-GAN checkpoint when possible.

Old training runs only saved ``best_generator.pth`` (the best-PSNR epoch) and
``last_generator.pth`` (final epoch). They did NOT save per-epoch snapshots.
That means we usually cannot recover an arbitrary historical epoch's weights.

But there is one case where we can: if the **post-GAN SSIM peak** happens to
fall on either the **best-PSNR epoch** (already on disk as
``best_generator.pth``) or the **final epoch** (already on disk as
``last_generator.pth``), we can promote that file to a companion
``best_ssim_generator.pth`` plus ``best_ssim_metrics.json`` without retraining.

For BACH ESRGAN this happens to be true: the post-GAN SSIM peak is at the same
epoch as the best PSNR. For brain-MRI ESRGAN you should just train using the
updated ``train.py`` — it now saves the SSIM checkpoint live.

Usage
-----
    python src/recover_best_ssim.py --run-dir runs/esrgan_bach_x4
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def _read_log(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                row = {
                    "epoch": int(r["epoch"]),
                    "val_psnr": float(r["val_psnr"]) if r.get("val_psnr") not in ("", None) else None,
                    "val_ssim": float(r["val_ssim"]) if r.get("val_ssim") not in ("", None) else None,
                }
            except Exception:
                continue
            rows.append(row)
    return [r for r in rows if r["val_ssim"] is not None]


def _gan_start_epoch(used_config_path: Path) -> int:
    if not used_config_path.exists():
        return 0
    with used_config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return int((cfg.get("training", {}) or {}).get("gan_start_epoch", 0))


def _best_post_gan_ssim(rows: List[Dict[str, Any]], gan_start: int) -> Tuple[int | None, float]:
    candidates = [r for r in rows if r["epoch"] >= gan_start]
    if not candidates:
        return None, -1.0
    best = max(candidates, key=lambda r: r["val_ssim"])
    return best["epoch"], best["val_ssim"]


def _saved_best_epoch(best_metrics_path: Path) -> int | None:
    if not best_metrics_path.exists():
        return None
    try:
        d = json.loads(best_metrics_path.read_text(encoding="utf-8"))
        return int(d.get("epoch")) if d.get("epoch") is not None else None
    except Exception:
        return None


def recover(run_dir: Path) -> int:
    csv_path = run_dir / "train_log.csv"
    used_cfg_path = run_dir / "used_config.yaml"
    ckpt_dir = run_dir / "checkpoints"
    best_metrics_path = run_dir / "best_metrics.json"

    if not csv_path.exists():
        print(f"FAIL: missing {csv_path}")
        return 2

    rows = _read_log(csv_path)
    if not rows:
        print(f"FAIL: no validated rows in {csv_path}")
        return 2

    gan_start = _gan_start_epoch(used_cfg_path)
    target_epoch, target_ssim = _best_post_gan_ssim(rows, gan_start)

    print(f"Run dir:           {run_dir}")
    print(f"gan_start_epoch:   {gan_start}")
    print(f"Total epochs in log: {rows[-1]['epoch']}")
    if target_epoch is None:
        print("FAIL: no post-GAN rows in log — can't pick a target. Train past gan_start_epoch first.")
        return 1

    print(f"Best post-GAN SSIM:  epoch={target_epoch}, ssim={target_ssim:.4f}")

    saved_best_epoch = _saved_best_epoch(best_metrics_path)
    last_epoch = rows[-1]["epoch"]

    src_path: Path | None = None
    src_label = ""
    if saved_best_epoch is not None and saved_best_epoch == target_epoch:
        src_path = ckpt_dir / "best_generator.pth"
        src_label = f"best_generator.pth (epoch {saved_best_epoch})"
    elif last_epoch == target_epoch:
        src_path = ckpt_dir / "last_generator.pth"
        src_label = f"last_generator.pth (epoch {last_epoch})"

    if src_path is None or not src_path.exists():
        print()
        print("Cannot recover automatically:")
        print(f"  target post-GAN-SSIM epoch is {target_epoch};")
        print(f"  best_generator.pth corresponds to epoch {saved_best_epoch};")
        print(f"  last_generator.pth corresponds to epoch {last_epoch}.")
        print("  Neither matches. Per-epoch checkpoints were not saved.")
        print()
        print("Honest options:")
        print("  1) Resume training for a few epochs (the new train.py will save best_ssim live)")
        print("  2) Retrain from scratch")
        print("  3) Disclose in the thesis that the demoed checkpoint is selected on PSNR")
        return 1

    # --- Recover ---
    # Find the matching log row to populate metrics JSON.
    matched_row = next(r for r in rows if r["epoch"] == target_epoch)
    psnr_at_target = matched_row.get("val_psnr")

    dst_pth = ckpt_dir / "best_ssim_generator.pth"
    dst_json = run_dir / "best_ssim_metrics.json"
    shutil.copy2(src_path, dst_pth)
    payload = {
        "epoch": target_epoch,
        "val_psnr": psnr_at_target,
        "val_ssim": target_ssim,
        "model_name": "esrgan",
        "selection": "best_post_gan_ssim",
        "gan_start_epoch": gan_start,
        "recovered_from": src_label,
        "note": (
            "Recovered by src/recover_best_ssim.py because the post-GAN-SSIM peak "
            "happens to coincide with a checkpoint that was actually saved to disk. "
            "Per-epoch snapshots were not available."
        ),
    }
    dst_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"OK: recovered post-GAN-SSIM checkpoint")
    print(f"  source:  {src_label}")
    print(f"  copied to: {dst_pth}")
    print(f"  metrics:   {dst_json}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Recover best-SSIM-post-GAN checkpoint where possible.")
    p.add_argument("--run-dir", required=True, help="e.g. runs/esrgan_bach_x4")
    args = p.parse_args()
    raise SystemExit(recover(Path(args.run_dir)))


if __name__ == "__main__":
    main()
