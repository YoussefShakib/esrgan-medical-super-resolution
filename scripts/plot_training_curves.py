"""Generate training curve comparison: original ESRGAN vs stable-GAN ESRGAN."""

import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_log(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if v != "" else None for k, v in row.items()})
    return rows

orig = load_log(PROJECT_ROOT / "runs/esrgan_bach_x4/train_log.csv")
stab = load_log(PROJECT_ROOT / "runs/esrgan_stable_v1/train_log.csv")

def col(rows, key):
    return [r["epoch"] for r in rows if r.get(key) is not None], \
           [r[key]    for r in rows if r.get(key) is not None]

GAN_START_ORIG = 15
GAN_START_STAB = 15

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("ESRGAN Training: Original vs Stable-GAN Schedule", fontsize=13, fontweight="bold")

# ── Discriminator loss ─────────────────────────────────────────────────────────
ax = axes[0]
ox, oy = col(orig, "d_loss")
sx, sy = col(stab, "d_loss")
ax.plot(ox, oy, color="#F44336", alpha=0.85, linewidth=1.4, label="Original")
ax.plot(sx, sy, color="#2196F3", alpha=0.85, linewidth=1.4, label="Stable-GAN")
ax.axvline(GAN_START_ORIG, color="#F44336", linestyle="--", alpha=0.4, linewidth=1)
ax.axvline(GAN_START_STAB, color="#2196F3", linestyle="--", alpha=0.4, linewidth=1)
ax.set_title("Discriminator Loss\n(lower & stable = better)", fontsize=10)
ax.set_xlabel("Epoch")
ax.set_ylabel("D Loss")
ax.set_ylim(-0.05, 2.5)
ax.legend(fontsize=9)
# annotate crash
ax.annotate("D saturates\n(0.25 by ep 18)", xy=(18, 0.25), xytext=(28, 0.6),
            fontsize=8, color="#F44336",
            arrowprops=dict(arrowstyle="->", color="#F44336", lw=1))
ax.annotate("Stable: 1.4–1.9\nthroughout", xy=(45, 1.6), xytext=(50, 2.1),
            fontsize=8, color="#2196F3",
            arrowprops=dict(arrowstyle="->", color="#2196F3", lw=1))

# ── Val PSNR ───────────────────────────────────────────────────────────────────
ax = axes[1]
ox, oy = col(orig, "val_psnr")
sx, sy = col(stab, "val_psnr")
ax.plot(ox, oy, color="#F44336", alpha=0.85, linewidth=1.4, label="Original")
ax.plot(sx, sy, color="#2196F3", alpha=0.85, linewidth=1.4, label="Stable-GAN")
ax.axvline(GAN_START_ORIG, color="#888", linestyle="--", alpha=0.5, linewidth=1,
           label="GAN start (ep 15)")
best_orig_psnr = max(oy)
best_stab_psnr = max(sy)
best_orig_ep   = ox[oy.index(best_orig_psnr)]
best_stab_ep   = sx[sy.index(best_stab_psnr)]
ax.scatter([best_orig_ep], [best_orig_psnr], color="#F44336", zorder=5, s=60,
           label=f"Orig best: {best_orig_psnr:.2f} dB (ep {int(best_orig_ep)})")
ax.scatter([best_stab_ep], [best_stab_psnr], color="#2196F3", zorder=5, s=60,
           label=f"Stable best: {best_stab_psnr:.2f} dB (ep {int(best_stab_ep)})")
ax.set_title("Val PSNR (dB)\n(higher = better)", fontsize=10)
ax.set_xlabel("Epoch")
ax.set_ylabel("PSNR (dB)")
ax.set_ylim(18, 31)
ax.legend(fontsize=8)

# ── Val SSIM ───────────────────────────────────────────────────────────────────
ax = axes[2]
ox, oy = col(orig, "val_ssim")
sx, sy = col(stab, "val_ssim")
ax.plot(ox, oy, color="#F44336", alpha=0.85, linewidth=1.4, label="Original")
ax.plot(sx, sy, color="#2196F3", alpha=0.85, linewidth=1.4, label="Stable-GAN")
ax.axvline(GAN_START_ORIG, color="#888", linestyle="--", alpha=0.5, linewidth=1,
           label="GAN start (ep 15)")
best_orig_ssim = max(oy)
best_stab_ssim = max(sy)
best_orig_ep2  = ox[oy.index(best_orig_ssim)]
best_stab_ep2  = sx[sy.index(best_stab_ssim)]
ax.scatter([best_orig_ep2], [best_orig_ssim], color="#F44336", zorder=5, s=60,
           label=f"Orig best: {best_orig_ssim:.4f} (ep {int(best_orig_ep2)})")
ax.scatter([best_stab_ep2], [best_stab_ssim], color="#2196F3", zorder=5, s=60,
           label=f"Stable best: {best_stab_ssim:.4f} (ep {int(best_stab_ep2)})")
ax.set_title("Val SSIM\n(higher = better)", fontsize=10)
ax.set_xlabel("Epoch")
ax.set_ylabel("SSIM")
ax.set_ylim(0.5, 0.86)
ax.legend(fontsize=8)

plt.tight_layout()
out = PROJECT_ROOT / "runs/esrgan_stable_v1/training_curve_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# Print summary table
print("\nKey numbers:")
print(f"  Original  — best PSNR: {best_orig_psnr:.2f} dB @ ep {int(best_orig_ep)}, "
      f"best SSIM: {best_orig_ssim:.4f} @ ep {int(best_orig_ep2)}")
print(f"  Stable    — best PSNR: {best_stab_psnr:.2f} dB @ ep {int(best_stab_ep)}, "
      f"best SSIM: {best_stab_ssim:.4f} @ ep {int(best_stab_ep2)}")
print(f"  Stable d_loss range (GAN phase): "
      f"{min(r['d_loss'] for r in stab if r['d_loss'] and r['d_loss'] > 0):.2f} – "
      f"{max(r['d_loss'] for r in stab if r['d_loss']):.2f}")
print(f"  Original d_loss min (GAN phase): "
      f"{min(r['d_loss'] for r in orig if r['d_loss'] and r['d_loss'] > 0):.2f}")
