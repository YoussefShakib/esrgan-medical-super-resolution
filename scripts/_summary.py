"""Print consolidated defence numbers summary."""
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

print()
print("="*70)
print("  CONSOLIDATED DEFENCE NUMBERS")
print("="*70)

# ── BACH Benchmark ────────────────────────────────────────────────────────────
print()
print("STEP 1 -- BACH BENCHMARK (n=60, val_HR_stratified, eval_crop=512)")
csv_path = ROOT / "results/bach_benchmark_lpips.csv"
rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))

valid_lpips = [(r["Method"], float(r["LPIPS"]))    for r in rows if r["LPIPS"]   not in ("MISSING", "")]
valid_psnr  = [(r["Method"], float(r["PSNR_dB"])) for r in rows if r["PSNR_dB"] not in ("MISSING", "")]
best_lpips_method = min(valid_lpips, key=lambda x: x[1])[0] if valid_lpips else None
best_psnr_method  = max(valid_psnr,  key=lambda x: x[1])[0] if valid_psnr  else None

print(f"  {'Method':<18} {'PSNR(dB)':>9} {'SSIM':>8} {'LPIPS':>8}")
print("  " + "-"*48)
for row in rows:
    psnr_s  = f"{float(row['PSNR_dB']):.4f}" if row["PSNR_dB"] not in ("MISSING","") else "MISSING "
    ssim_s  = f"{float(row['SSIM']):.4f}"    if row["SSIM"]    not in ("MISSING","") else "MISSING "
    lpips_s = f"{float(row['LPIPS']):.4f}"   if row["LPIPS"]   not in ("MISSING","") else "MISSING "
    tags = []
    if row["Method"] == best_psnr_method:  tags.append("<-- best PSNR")
    if row["Method"] == best_lpips_method: tags.append("<-- best LPIPS")
    tag = "  " + "  ".join(tags) if tags else ""
    print(f"  {row['Method']:<18} {psnr_s:>8} {ssim_s:>8} {lpips_s:>8}{tag}")

# ── MRI Benchmark ─────────────────────────────────────────────────────────────
print()
print("STEP 5 -- BRAIN MRI (n=180, eval_crop=256, best_generator.pth = epoch 10)")
mri_csv = ROOT / "results/mri_esrgan_vs_bicubic.csv"
print(f"  {'Method':<18} {'PSNR(dB)':>9} {'SSIM':>8} {'LPIPS':>8}")
print("  " + "-"*48)
for row in csv.DictReader(open(mri_csv, encoding="utf-8")):
    psnr_s  = f"{float(row['PSNR_dB']):.4f}" if row.get("PSNR_dB") not in (None,"MISSING","") else "MISSING "
    ssim_s  = f"{float(row['SSIM']):.4f}"    if row.get("SSIM")    not in (None,"MISSING","") else "MISSING "
    lpips_s = f"{float(row['LPIPS']):.4f}"   if row.get("LPIPS")   not in (None,"MISSING","") else "MISSING "
    print(f"  {row['Method']:<18} {psnr_s:>8} {ssim_s:>8} {lpips_s:>8}")

# ── SR -> Classification ───────────────────────────────────────────────────────
print()
print("STEP 4 -- SR -> CLASSIFICATION STUDY")
data = json.loads((ROOT / "runs/sr_classification_study.json").read_text(encoding="utf-8"))
for ds, res in data.items():
    arch = res.get("classifier_arch", "?")
    n    = res.get("n_samples", "?")
    print(f"  {ds.upper()}  (arch={arch}, n={n})")
    for ck, cond in res.get("conditions", {}).items():
        m = cond["metrics"]
        label = cond["label"]
        print(f"    {ck:<22}  top1={m['top1_acc']:.4f}  macro_f1={m['macro_f1']:.4f}  [{label}]")
    print(f"  Finding: {res.get('finding', '-')}")
    print()

# ── Deliverable checklist ─────────────────────────────────────────────────────
print("DELIVERABLE CHECKLIST")
print("-"*70)
deliverables = [
    ("results/bach_benchmark_lpips.csv",        "Step 1 -- BACH LPIPS CSV"),
    ("figures/bach_benchmark_table.png",         "Step 1 -- BACH benchmark table figure"),
    ("figures/perception_distortion_scatter.png","Step 2 -- P-D scatter"),
    ("figures/qualitative_zoom_invasive.png",    "Step 3 -- Qualitative zoom panel"),
    ("results/sr_classification_study.csv",      "Step 4 -- SR->class CSV"),
    ("figures/sr_classification_bach.png",       "Step 4 -- BACH bar chart"),
    ("figures/sr_classification_brisc.png",      "Step 4 -- BRISC bar chart"),
    ("results/mri_esrgan_vs_bicubic.csv",        "Step 5 -- MRI LPIPS CSV"),
    ("figures/stabilisation_curves_clean.png",   "Step 6 -- Stabilisation curves"),
    ("figures/demo/demo_before_after.png",       "Step 7 -- Demo before/after"),
    ("figures/demo/demo_change_map.png",         "Step 7 -- Demo change map"),
    ("figures/demo/demo_nuclei_roi.png",         "Step 7 -- Demo nuclei ROI"),
]
all_ok = True
for rel, label in deliverables:
    p = ROOT / rel
    status = "OK     " if p.exists() else "MISSING"
    if not p.exists(): all_ok = False
    print(f"  [{status}]  {label}")

print()
if all_ok:
    print("  All 12 deliverables present.")
else:
    print("  *** Some deliverables are MISSING ***")
print("="*70)
