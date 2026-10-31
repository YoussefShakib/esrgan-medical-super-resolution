"""Write sr_classification_study.csv and bar charts from the saved JSON."""
import csv, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)

PETROL = "#103B3C"; TEAL = "#2E9B8F"; GOLD = "#E0A82E"; GREY = "#6B6B6B"

data = json.loads((ROOT / "runs/sr_classification_study.json").read_text(encoding="utf-8"))

# ── CSV ───────────────────────────────────────────────────────────────────────
rows = [["dataset", "classifier_arch", "n_samples", "condition", "label",
         "top1_acc", "macro_f1"]]
for ds, res in data.items():
    arch = res.get("classifier_arch", "?")
    n    = res.get("n_samples", "?")
    for ck, cond in res.get("conditions", {}).items():
        m = cond["metrics"]
        rows.append([ds, arch, n, ck, cond["label"],
                     f"{m['top1_acc']:.4f}", f"{m['macro_f1']:.4f}"])

csv_path = RESULTS / "sr_classification_study.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(rows)
print(f"CSV saved: {csv_path}")

# ── Bar charts ────────────────────────────────────────────────────────────────
for ds_name, res in data.items():
    conditions = res.get("conditions", {})
    cond_keys  = list(conditions.keys())
    cond_labels = ["HR (GT)", "LR bicubic", "ESRGAN SR"][:len(cond_keys)]
    arch = res.get("classifier_arch", "?")
    n    = res.get("n_samples", "?")

    metrics_to_show = [("top1_acc", "Top-1 Accuracy"), ("macro_f1", "Macro F1")]
    bar_colors = [TEAL, GOLD]
    x = np.arange(len(cond_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    for m_idx, (metric, label) in enumerate(metrics_to_show):
        vals = [conditions[c]["metrics"][metric] for c in cond_keys]
        bars = ax.bar(x + m_idx * width, vals, width,
                      label=label, color=bar_colors[m_idx], alpha=0.88)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    ax.set_xticks(x + width/2)
    ax.set_xticklabels(cond_labels, rotation=12, ha="right", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title(f"SR -> Classification Study: {ds_name.upper()}\n"
                 f"classifier={arch}  n={n}",
                 fontsize=13, fontweight="bold", color=PETROL)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    finding = res.get("finding", "")
    ax.text(0.5, -0.24, finding, transform=ax.transAxes,
            ha="center", va="top", fontsize=9, style="italic", color=GREY, wrap=True)
    fig.tight_layout()
    out = FIGURES / f"sr_classification_{ds_name}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure saved: {out}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SR -> Classification Study  (full run)")
print("="*65)
for ds, res in data.items():
    arch = res.get("classifier_arch", "?")
    n    = res.get("n_samples", "?")
    print(f"\n  {ds.upper()}  (arch={arch}, n={n})")
    for ck, cond in res.get("conditions", {}).items():
        m = cond["metrics"]
        print(f"    {ck:<20}  top1={m['top1_acc']:.4f}  macro_f1={m['macro_f1']:.4f}")
    print(f"  Finding: {res.get('finding', '-')}")
