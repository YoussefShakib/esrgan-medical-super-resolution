"""Train ESRGAN on every medical domain back-to-back (or in parallel).

Sequential mode (default): runs each config one after the other; each one's
``resume: true`` means re-running picks up where it left off. Safe on a single
GPU or on CPU.

Parallel mode (--parallel): launches both as background subprocesses.
**Only** use this if you have two GPUs (or are on CPU and accept halved
throughput); otherwise the ESRGAN runs will OOM each other.

Logs go to ``runs/<experiment_name>/launcher_shell.log`` so you can tail them
live.

Usage
-----
    python scripts/train_all_esrgan.py
    python scripts/train_all_esrgan.py --parallel
    python scripts/train_all_esrgan.py --only bach
    python scripts/train_all_esrgan.py --only brain_mri
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOMAIN_CONFIGS: Dict[str, str] = {
    "bach": "configs/esrgan_bach_medical.yaml",
    "brain_mri": "configs/esrgan_brain_mri.yaml",
}


def _experiment_name(config_path: Path) -> str:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    name = cfg.get("experiment_name")
    if not name:
        raise RuntimeError(f"experiment_name missing in {config_path}")
    save_dir = cfg.get("save_dir", "runs")
    return f"{save_dir}/{name}"


def _run_one(config_path: Path, log_path: Path, env: Dict[str, str]) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    log_handle.write(f"\n\n========== {time.strftime('%Y-%m-%d %H:%M:%S')} ==========\n")
    log_handle.flush()
    cmd = [sys.executable, "src/train.py", "--config", str(config_path)]
    print(f"[launch] {' '.join(cmd)}  (log: {log_path})")
    return subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
    )


def _print_best(run_dir: Path, label: str) -> None:
    best_psnr = run_dir / "best_metrics.json"
    best_ssim = run_dir / "best_ssim_metrics.json"
    print(f"\n[summary] {label} ({run_dir.name}):")
    for tag, p in (("best PSNR ", best_psnr), ("best SSIM ", best_ssim)):
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                print(
                    f"  {tag} epoch={d.get('epoch')}  "
                    f"PSNR={d.get('val_psnr'):.3f}  "
                    f"SSIM={d.get('val_ssim'):.4f}"
                )
            except Exception as e:
                print(f"  {tag} could not parse: {e}")
        else:
            print(f"  {tag} (not yet produced)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ESRGAN across all medical domains.")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run domains concurrently (only safe with multiple GPUs / CPU-only).",
    )
    parser.add_argument(
        "--only",
        choices=list(DOMAIN_CONFIGS),
        default=None,
        help="Train only one domain.",
    )
    parser.add_argument(
        "--gpu-map",
        nargs="+",
        default=None,
        metavar="DOMAIN=ID",
        help=(
            "Pin domains to GPUs in --parallel mode, e.g. "
            "--gpu-map bach=0 brain_mri=1. Sets CUDA_VISIBLE_DEVICES for each subprocess."
        ),
    )
    args = parser.parse_args()

    selected = [args.only] if args.only else list(DOMAIN_CONFIGS)
    gpu_map: Dict[str, str] = {}
    if args.gpu_map:
        for entry in args.gpu_map:
            if "=" not in entry:
                parser.error(f"--gpu-map entries must be DOMAIN=ID, got {entry!r}")
            k, v = entry.split("=", 1)
            if k not in DOMAIN_CONFIGS:
                parser.error(f"--gpu-map unknown domain {k!r}; valid: {list(DOMAIN_CONFIGS)}")
            gpu_map[k] = v

    runs: List[Dict[str, Path]] = []
    for domain in selected:
        cfg_path = PROJECT_ROOT / DOMAIN_CONFIGS[domain]
        run_dir_str = _experiment_name(cfg_path)
        run_dir = PROJECT_ROOT / run_dir_str
        log_path = run_dir / "launcher_shell.log"
        runs.append({"domain": domain, "config": cfg_path, "run_dir": run_dir, "log": log_path})

    print(f"Mode: {'parallel' if args.parallel else 'sequential'}")
    print(f"Domains: {[r['domain'] for r in runs]}")
    if args.parallel and len(runs) > 1:
        print(
            "WARNING: parallel mode will OOM if both runs share one GPU. "
            "Use --gpu-map or skip --parallel on single-GPU hardware."
        )

    overall_start = time.time()
    failures: List[str] = []

    if args.parallel:
        procs: List[Dict] = []
        for r in runs:
            env = os.environ.copy()
            if r["domain"] in gpu_map:
                env["CUDA_VISIBLE_DEVICES"] = gpu_map[r["domain"]]
            p = _run_one(r["config"], r["log"], env)
            procs.append({"proc": p, "info": r})
        for entry in procs:
            rc = entry["proc"].wait()
            if rc != 0:
                failures.append(entry["info"]["domain"])
                print(f"[fail] {entry['info']['domain']} exited with code {rc}")
            else:
                print(f"[done] {entry['info']['domain']}")
    else:
        for r in runs:
            env = os.environ.copy()
            if r["domain"] in gpu_map:
                env["CUDA_VISIBLE_DEVICES"] = gpu_map[r["domain"]]
            p = _run_one(r["config"], r["log"], env)
            rc = p.wait()
            if rc != 0:
                failures.append(r["domain"])
                print(f"[fail] {r['domain']} exited with code {rc}")
                # Stop on first failure in sequential mode — typical expectation.
                break
            print(f"[done] {r['domain']}")

    print(f"\nTotal wall time: {(time.time() - overall_start) / 60:.1f} min")

    for r in runs:
        if r["domain"] not in failures:
            _print_best(r["run_dir"], r["domain"])

    if failures:
        print(f"\nFAILED: {failures}")
        return 1
    print("\nAll requested domains finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
