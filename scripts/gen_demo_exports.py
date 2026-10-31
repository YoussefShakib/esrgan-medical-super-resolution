"""
Generate demo before/after panel and change-map overlay for defence slides.
Saves to figures/demo/ at 300 DPI, bypassing Streamlit.

Uses the first Invasive BACH val image (same as qualitative_zoom_invasive.png).
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _d in [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from src.models import build_model

DEMO_DIR = PROJECT_ROOT / "figures" / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

SCALE   = 4
PALETTE = {"petrol": "#103B3C", "cream": "#F3EFE4", "terra": "#C15B43", "teal": "#2E9B8F"}
CKPT    = "runs/esrgan_bach_x4/checkpoints/best_generator.pth"

device = torch.device("cpu")

# ── load model ────────────────────────────────────────────────────────────────
model = build_model("esrgan", SCALE).to(device)
ckpt  = torch.load(PROJECT_ROOT / CKPT, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"], strict=False)
model.eval()
print(f"Loaded: {CKPT}")

# ── load image ────────────────────────────────────────────────────────────────
val_dir = PROJECT_ROOT / "datasets" / "val_HR_stratified"
invasive = sorted([p for p in val_dir.iterdir()
                   if p.stem.lower().startswith("invasive")])[0]
print(f"Image : {invasive.name}")

hr_full = Image.open(invasive).convert("RGB")
w, h    = hr_full.size
side    = min(w, h, 512)
left    = (w - side) // 2
top     = (h - side) // 2
hr_img  = hr_full.crop((left, top, left + side, top + side))
side    = side - (side % SCALE)
hr_img  = hr_img.crop((0, 0, side, side))

lr_img  = hr_img.resize((side // SCALE, side // SCALE), Image.BICUBIC)

lr_t  = TF.to_tensor(lr_img).unsqueeze(0)
hr_t  = TF.to_tensor(hr_img).unsqueeze(0)

with torch.no_grad():
    sr_t = model(lr_t).clamp(0, 1)

bic_t = TF.resize(lr_t,
                   size=[side, side],
                   interpolation=TF.InterpolationMode.BICUBIC,
                   antialias=True).clamp(0, 1)

def _to_np(t): return (t[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
hr_np  = _to_np(hr_t)
bic_np = _to_np(bic_t)
sr_np  = _to_np(sr_t)

# ─────────────────────────────────────────────────────────────────────────────
# Figure A: Before / After panel  (LR bicubic | HR ground truth | ESRGAN SR)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
fig.patch.set_facecolor("white")
fig.suptitle(f"SR Demo — Invasive BACH  ({invasive.stem})",
             fontsize=14, fontweight="bold", color=PALETTE["petrol"])

for ax, img, title, color in zip(
    axes,
    [bic_np, hr_np, sr_np],
    ["LR Bicubic\n(input after ÷4×4)", "HR Ground Truth", "ESRGAN ×4 SR\n(research prototype)"],
    ["#888888", PALETTE["teal"], PALETTE["terra"]],
):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=12, color=color, fontweight="bold", pad=6)

plt.tight_layout(rect=[0, 0, 1, 0.93])
out_a = DEMO_DIR / "demo_before_after.png"
fig.savefig(out_a, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_a}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure B: Change-map overlay  (ESRGAN SR − bicubic, scaled difference)
# ─────────────────────────────────────────────────────────────────────────────
diff = sr_np.astype(np.int16) - bic_np.astype(np.int16)
diff_magnitude = np.abs(diff).mean(axis=2)   # (H, W)
diff_norm = diff_magnitude / (diff_magnitude.max() + 1e-8)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5.5))
fig.patch.set_facecolor("white")
fig.suptitle("SR Change Map: ESRGAN − Bicubic Baseline",
             fontsize=14, fontweight="bold", color=PALETTE["petrol"])

ax1.imshow(bic_np); ax1.axis("off")
ax1.set_title("Bicubic (baseline)", fontsize=12, color="#888888", fontweight="bold")

ax2.imshow(sr_np);  ax2.axis("off")
ax2.set_title("ESRGAN SR", fontsize=12, color=PALETTE["terra"], fontweight="bold")

im = ax3.imshow(diff_norm, cmap="hot", vmin=0, vmax=1)
ax3.axis("off")
ax3.set_title("Change Map\n(bright = more SR detail added)",
              fontsize=12, color=PALETTE["petrol"], fontweight="bold")
cb = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
cb.ax.tick_params(labelsize=9)
cb.set_label("Relative change intensity", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.93])
out_b = DEMO_DIR / "demo_change_map.png"
fig.savefig(out_b, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_b}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure C: Zoomed-in comparison (HR vs SR, side-by-side, nuclei ROI)
# ─────────────────────────────────────────────────────────────────────────────
roi_size = 128
rx = (side - roi_size) // 2
ry = (side - roi_size) // 2

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
fig.patch.set_facecolor("white")
fig.suptitle(f"Nuclei ROI Detail ({roi_size}×{roi_size} px, centre crop)",
             fontsize=13, fontweight="bold", color=PALETTE["petrol"])

for ax, img, title, col in zip(
    axes,
    [bic_np[ry:ry+roi_size, rx:rx+roi_size],
     hr_np[ry:ry+roi_size, rx:rx+roi_size],
     sr_np[ry:ry+roi_size, rx:rx+roi_size]],
    ["Bicubic (baseline)", "HR Ground Truth", "ESRGAN SR"],
    ["#888888", PALETTE["teal"], PALETTE["terra"]],
):
    ax.imshow(img, interpolation="nearest")
    ax.axis("off")
    ax.set_title(title, fontsize=12, color=col, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.90])
out_c = DEMO_DIR / "demo_nuclei_roi.png"
fig.savefig(out_c, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_c}")

print("\nDemo exports complete. Files in figures/demo/:")
for f in sorted(DEMO_DIR.iterdir()):
    print(f"  {f.name}")
