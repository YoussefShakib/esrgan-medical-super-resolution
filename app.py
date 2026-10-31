"""ESRGAN-Based Super-Resolution for Medical Images — thesis research demo."""

from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison

from app_services import (
    align_hr_for_scale,
    analyze_image,
    bicubic_downsample,
    bicubic_upsample,
    compare_analysis,
    compute_change_map,
    compute_detail_residual,
    compute_psnr,
    compute_ssim,
    create_hybrid_bicubic_esrgan,
    default_lr_crop_box,
    findings_to_markdown,
    generate_findings_report,
    load_sr_service,
    make_triple_crop_panel,
    resize_for_display,
    scale_crop_box,
    load_classifier_service,
)
from utils import set_seed


# ── CONFIG ────────────────────────────────────────────────────────────────────

AUTHOR_NAME  = "Youssef Shakib"
UNIVERSITY   = "German University in Cairo"
TAGLINE      = "Deep learning super-resolution for medical imaging"
DEMO_SEED    = 42

DEFAULT_CONFIG     = "configs/esrgan_ablation_mixed.yaml"
DEFAULT_CHECKPOINT = "runs/esrgan_ablation_mixed/checkpoints/best_ssim_generator.pth"
BEST_METRICS_PATH  = "runs/esrgan_ablation_mixed/best_ssim_metrics.json"

SCALE                  = 4
METRIC_CROP            = 4
MAX_HR_LONG_SIDE       = 4096
QUICK_PREVIEW_LONG_SIDE = 512
DEFAULT_TILE_SIZE      = 256
DEFAULT_TILE_OVERLAP   = 24
DEFAULT_CROP_LR        = 256
DEFAULT_ALPHA          = 0.8
ALPHA_BOOST_MAX        = 1.25

ACCENT    = "#4FC3B8"   # Primary teal (Healthcare primary)
ACCENT2   = "#34D399"   # Emerald green (health positive)
BG_BASE   = "#090D17"   # App base background
BG_CARD   = "#111827"   # Card surface
BG_CARD2  = "#0D1420"   # Deeper card variant
BORDER    = "#1E2A3B"   # Standard border
BORDER2   = "#263344"   # Slightly lighter border
TEXT      = "#F1F5F9"   # Primary text
TEXT2     = "#CBD5E1"   # Secondary text
CAPTION   = "#64748B"   # Muted caption
WARN_COL  = "#F59E0B"   # Amber warning
DANGER    = "#EF4444"   # Red danger

METHOD_HYBRID  = "Hybrid Bicubic + ESRGAN"
METHOD_ESRGAN  = "ESRGAN only"
METHOD_BICUBIC = "Bicubic only"

DOMAIN_BACH    = "Histopathology / BACH"
DOMAIN_BRAIN   = "Brain MRI"
DOMAIN_GENERAL = "General medical image"

DOMAIN_REGISTRY: Dict[str, Optional[Dict[str, str]]] = {
    DOMAIN_BACH: {
        "config":              "configs/esrgan_ablation_mixed.yaml",
        "checkpoint":          "runs/esrgan_ablation_mixed/checkpoints/best_ssim_generator.pth",
        "checkpoint_fallback": "runs/esrgan_ablation_mixed/checkpoints/best_ssim_generator.pth",
        "best_metrics":        "runs/esrgan_ablation_mixed/best_ssim_metrics.json",
        "best_metrics_fallback": "runs/esrgan_ablation_mixed/best_metrics.json",
        "label": "Stable-ESRGAN (BACH)",
    },
    DOMAIN_BRAIN: {
        "config":              "configs/esrgan_brain_mri.yaml",
        "checkpoint":          "runs/esrgan_brain_mri_x4/checkpoints/best_generator.pth",
        "checkpoint_fallback": "runs/esrgan_brain_mri_x4/checkpoints/best_generator.pth",
        "best_metrics":        "runs/esrgan_brain_mri_x4/best_metrics.json",
        "best_metrics_fallback": "runs/esrgan_brain_mri_x4/best_metrics.json",
        "label": "Stable-ESRGAN (Brain MRI)",
    },
    DOMAIN_GENERAL: None,
}

# Curated demo images — strongest per-domain examples from a validation scan
# (selected for the largest clean PSNR/SSIM/LPIPS margin over the bicubic
# baseline, not handpicked for looks).
SAMPLE_IMAGES: Dict[str, Optional[Dict[str, str]]] = {
    DOMAIN_BACH: {
        "path": "datasets/val_HR_stratified/Invasive_iv088.tif",
        "label": "Invasive carcinoma biopsy (BACH)",
        "talking_point": (
            "Stable-ESRGAN recovers visibly sharper nuclear and chromatin boundaries "
            "than bicubic upsampling on this invasive carcinoma biopsy — the clearest "
            "visible gap among the BACH validation images. SSIM rises from 0.452 to "
            "0.494 and LPIPS improves from 0.522 to 0.430, the largest perceptual "
            "(LPIPS) margin in the set, while never losing ground on PSNR or SSIM."
        ),
        "bicubic_psnr": 20.17, "bicubic_ssim": 0.452, "bicubic_lpips": 0.522,
        "esrgan_psnr": 20.26, "esrgan_ssim": 0.494, "esrgan_lpips": 0.430,
    },
    DOMAIN_BRAIN: {
        "path": "datasets/BRAIN_MRI/val_HR/glioma__brisc2025_train_00427_gl_co_t1.png",
        "label": "Glioma, coronal T1 (BRISC2025)",
        "talking_point": (
            "On this glioma MRI slice, Stable-ESRGAN clearly outperforms bicubic "
            "upsampling: PSNR improves by +2.11 dB (31.56 -> 33.67), SSIM by +0.014 "
            "(0.925 -> 0.939), and LPIPS drops from 0.176 to 0.102 — sharper tumor "
            "margins and tissue boundaries that are visibly softer in the bicubic baseline."
        ),
        "bicubic_psnr": 31.56, "bicubic_ssim": 0.925, "bicubic_lpips": 0.176,
        "esrgan_psnr": 33.67, "esrgan_ssim": 0.939, "esrgan_lpips": 0.102,
    },
}


def _resolve_checkpoint_pair(domain_entry: Dict[str, str]) -> Tuple[str, str]:
    """Return (checkpoint, metrics_json) — prefers SSIM pair when both files exist."""
    primary_ckpt    = domain_entry.get("checkpoint", "")
    primary_metrics = domain_entry.get("best_metrics", "")
    if (
        primary_ckpt and primary_metrics
        and Path(primary_ckpt).exists()
        and Path(primary_metrics).exists()
    ):
        return primary_ckpt, primary_metrics
    return (
        domain_entry.get("checkpoint_fallback", primary_ckpt),
        domain_entry.get("best_metrics_fallback", primary_metrics),
    )


# ── PAGE SETUP ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Stable-ESRGAN Medical Imaging",
    layout="wide",
    initial_sidebar_state="expanded",
)

set_seed(DEMO_SEED)

st.markdown(
    f"""
    <style>
      /* ── Google Fonts: Medical Clean (Figtree + Noto Sans) ── */
      @import url('https://fonts.googleapis.com/css2?family=Figtree:wght@300;400;500;600;700;800&family=Noto+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

      /* ── Global typography ───────────────────────────────── */
      html, body, [class*="css"], .stMarkdown, .stText,
      [data-testid="stSidebar"] {{ font-family: 'Noto Sans', system-ui, sans-serif; -webkit-font-smoothing: antialiased; }}
      h1, h2, h3, h4, h5, h6,
      .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {{
        font-family: 'Figtree', system-ui, sans-serif;
        letter-spacing: -0.025em; font-weight: 700;
      }}

      /* ── App background ──────────────────────────────────── */
      .stApp {{ background: {BG_BASE}; }}

      /* ── Layout ──────────────────────────────────────────── */
      .block-container {{
        padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1400px;
      }}

      /* ── Sidebar ─────────────────────────────────────────── */
      section[data-testid="stSidebar"] {{
        background: {BG_CARD2} !important;
        border-right: 1px solid {BORDER};
      }}
      section[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}

      /* ── Metrics ──────────────────────────────────────────── */
      [data-testid="stMetric"] {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        transition: border-color 0.2s;
      }}
      [data-testid="stMetric"]:hover {{ border-color: {BORDER2}; }}
      [data-testid="stMetricValue"] {{
        font-family: 'Figtree', sans-serif;
        font-size: 1.55rem; font-weight: 700; color: {ACCENT};
        letter-spacing: -0.03em;
      }}
      [data-testid="stMetricLabel"] {{
        font-family: 'Noto Sans', sans-serif;
        color: {CAPTION}; font-size: 0.71rem;
        text-transform: uppercase; letter-spacing: 0.09em; font-weight: 600;
      }}
      [data-testid="stMetricDelta"] {{ font-size: 0.8rem; font-weight: 600; }}

      /* ── Tabs ─────────────────────────────────────────────── */
      div[data-testid="stTabs"] {{
        border-bottom: 1px solid {BORDER};
        gap: 0.1rem;
      }}
      div[data-testid="stTabs"] button[role="tab"] {{
        font-family: 'Noto Sans', sans-serif;
        padding: 0.55rem 1rem; font-weight: 500; font-size: 0.85rem;
        color: {CAPTION}; border-radius: 6px 6px 0 0;
        transition: color 0.15s, background 0.15s;
      }}
      div[data-testid="stTabs"] button[role="tab"]:hover {{
        color: {TEXT2}; background: rgba(79,195,184,0.06);
      }}
      div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
        color: {ACCENT}; border-bottom: 2px solid {ACCENT}; font-weight: 600;
      }}

      /* ── Primary button ──────────────────────────────────── */
      .stButton > button[kind="primary"],
      .stDownloadButton > button[kind="primary"] {{
        font-family: 'Figtree', sans-serif;
        background: linear-gradient(135deg, #0891B2 0%, {ACCENT} 100%);
        color: #041017; border: none; font-weight: 700;
        letter-spacing: 0.01em; border-radius: 8px;
        box-shadow: 0 2px 10px rgba(79,195,184,0.22);
        transition: box-shadow 0.2s, transform 0.15s;
      }}
      .stButton > button[kind="primary"]:hover,
      .stDownloadButton > button[kind="primary"]:hover {{
        box-shadow: 0 4px 20px rgba(79,195,184,0.38);
        transform: translateY(-1px);
      }}
      .stButton > button[kind="secondary"],
      .stDownloadButton > button[kind="secondary"] {{
        font-family: 'Noto Sans', sans-serif;
        border: 1px solid {BORDER2}; border-radius: 8px;
        transition: border-color 0.15s, color 0.15s;
      }}
      .stButton > button[kind="secondary"]:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}

      /* ── Cards ────────────────────────────────────────────── */
      .esr-card {{
        background: linear-gradient(135deg, {BG_CARD} 0%, {BG_CARD2} 100%);
        border: 1px solid {BORDER}; border-left: 3px solid {ACCENT};
        border-radius: 10px; padding: 1rem 1.15rem;
        margin: 0.35rem 0 0.8rem 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.35);
      }}
      .esr-card-warn {{
        background: linear-gradient(135deg, #130f04 0%, #0f0c02 100%);
        border: 1px solid #2a2200; border-left: 3px solid {WARN_COL};
        border-radius: 10px; padding: 1rem 1.15rem; margin: 0.35rem 0 0.8rem 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.35);
      }}
      .esr-card small {{
        color: {CAPTION}; letter-spacing: 0.06em;
        text-transform: uppercase; font-size: 0.7rem; font-weight: 600;
      }}
      .esr-kv     {{ color: {TEXT};    font-family: 'Figtree', sans-serif; font-weight: 600; font-size: 1.05rem; letter-spacing: -0.01em; }}
      .esr-caption {{ color: {TEXT2};  font-size: 0.83rem; line-height: 1.6; }}

      /* ── Tags ─────────────────────────────────────────────── */
      .esr-tag {{
        display: inline-flex; align-items: center;
        padding: 0.12rem 0.55rem;
        background: rgba(79,195,184,0.08);
        border: 1px solid {ACCENT}; color: {ACCENT}; border-radius: 999px;
        font-size: 0.65rem; letter-spacing: 0.09em;
        text-transform: uppercase; font-weight: 700;
        font-family: 'Figtree', sans-serif;
      }}
      .esr-tag-warn {{
        display: inline-flex; align-items: center;
        padding: 0.12rem 0.55rem;
        background: rgba(245,158,11,0.08);
        border: 1px solid {WARN_COL}; color: {WARN_COL}; border-radius: 999px;
        font-size: 0.65rem; letter-spacing: 0.09em;
        text-transform: uppercase; font-weight: 700;
        font-family: 'Figtree', sans-serif;
      }}

      /* ── Step list ────────────────────────────────────────── */
      .esr-step {{
        display: flex; align-items: flex-start; gap: 0.6rem;
        padding: 0.42rem 0; color: {TEXT2}; font-size: 0.87rem;
        border-bottom: 1px solid {BORDER};
      }}
      .esr-step:last-child {{ border-bottom: none; }}
      .esr-step-num {{
        display: inline-flex; align-items: center; justify-content: center;
        min-width: 1.5rem; height: 1.5rem; border-radius: 50%;
        background: rgba(79,195,184,0.12); border: 1px solid {ACCENT};
        color: {ACCENT}; font-family: 'Figtree', sans-serif;
        font-size: 0.68rem; font-weight: 800; flex-shrink: 0; margin-top: 0.05rem;
      }}

      /* ── Placeholder ──────────────────────────────────────── */
      .esr-placeholder {{
        background: {BG_CARD}; border: 1px dashed {BORDER2};
        border-radius: 12px; padding: 3.5rem 1rem;
        text-align: center; color: {CAPTION};
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
      }}
      .esr-placeholder-icon {{
        font-size: 2.4rem; margin-bottom: 0.6rem; opacity: 0.7;
        display: block;
      }}
      .esr-placeholder-msg {{
        font-family: 'Figtree', sans-serif; font-size: 0.95rem;
        font-weight: 600; color: {TEXT2}; margin-bottom: 0.25rem;
      }}
      .esr-placeholder-hint {{ font-size: 0.8rem; color: {CAPTION}; }}

      /* ── Delta colours ────────────────────────────────────── */
      .d-pos {{ color: {ACCENT2};  font-weight: 600; }}
      .d-neg {{ color: #F87171;   font-weight: 600; }}
      .d-neu {{ color: {CAPTION}; font-weight: 600; }}

      /* ── Expander ─────────────────────────────────────────── */
      [data-testid="stExpander"] {{
        background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 10px;
      }}
      [data-testid="stExpander"] summary {{
        font-family: 'Noto Sans', sans-serif;
        font-weight: 500; font-size: 0.88rem; color: {TEXT2};
      }}
      [data-testid="stExpander"] summary:hover {{ color: {TEXT}; }}

      /* ── File uploader ────────────────────────────────────── */
      [data-testid="stFileUploader"] section {{
        border: 2px dashed {BORDER2}; border-radius: 10px;
        transition: border-color 0.2s, background 0.2s;
      }}
      [data-testid="stFileUploader"] section:hover {{
        border-color: {ACCENT}; background: rgba(79,195,184,0.04);
      }}

      /* ── Progress bar ─────────────────────────────────────── */
      [data-testid="stProgressBar"] > div > div {{
        background: linear-gradient(90deg, #0891B2, {ACCENT});
        border-radius: 999px;
      }}

      /* ── Dataframe ────────────────────────────────────────── */
      [data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}
      [data-testid="stDataFrameGlideDataEditor"] {{ font-family: 'Noto Sans', sans-serif; }}

      /* ── Alert/Info banners ───────────────────────────────── */
      .stAlert {{ border-radius: 10px; border-left-width: 3px; }}
      .stInfo  {{ border-left-color: {ACCENT}; }}

      /* ── Scrollbar ────────────────────────────────────────── */
      ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
      ::-webkit-scrollbar-track {{ background: {BG_BASE}; }}
      ::-webkit-scrollbar-thumb {{ background: {BORDER2}; border-radius: 3px; }}
      ::-webkit-scrollbar-thumb:hover {{ background: #374151; }}

      hr {{ border-color: {BORDER}; margin: 1.1rem 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ── CACHE ─────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_service(config_path: str, checkpoint_path: str):
    return load_sr_service(config_path, checkpoint_path)


@st.cache_resource(show_spinner=False)
def get_classifier_service(dataset_name: str):
    return load_classifier_service(dataset_name)


@st.cache_data(show_spinner=False)
def load_best_metrics(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ── UTILITIES ─────────────────────────────────────────────────────────────────

def _png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _delta_span(v: float, digits: int = 3, lower_is_better: bool = False) -> str:
    """Coloured ±value HTML span."""
    better = (v < 0) if lower_is_better else (v > 0)
    css = "d-pos" if better else ("d-neg" if v != 0 else "d-neu")
    sign = "+" if v >= 0 else ""
    return f"<span class='{css}'>{sign}{v:.{digits}f}</span>"


def _prepare_hr(image: Image.Image, mode: str) -> Tuple[Image.Image, str]:
    image = image.convert("RGB")
    w, h  = image.size
    notes: list[str] = []
    long  = max(w, h)
    if mode == "quick":
        if long > QUICK_PREVIEW_LONG_SIDE:
            s    = QUICK_PREVIEW_LONG_SIDE / float(long)
            new  = (max(SCALE, int(w * s)), max(SCALE, int(h * s)))
            image = image.resize(new, Image.LANCZOS)
            notes.append(f"Quick preview: resized {w}×{h} → {image.width}×{image.height}.")
        else:
            notes.append(f"Quick preview: input ≤ {QUICK_PREVIEW_LONG_SIDE}px, no resize needed.")
    else:
        if long > MAX_HR_LONG_SIDE:
            s    = MAX_HR_LONG_SIDE / float(long)
            new  = (int(w * s), int(h * s))
            image = image.resize(new, Image.LANCZOS)
            notes.append(f"Capped at {MAX_HR_LONG_SIDE}px: {w}×{h} → {image.width}×{image.height}.")
    image = align_hr_for_scale(image, SCALE)
    notes.append(f"Aligned to ×{SCALE}: {image.width}×{image.height}.")
    return image, " ".join(notes)


def _findings_mode(domain: str) -> str:
    return {"Histopathology / BACH": "histopathology", "Brain MRI": "brain_mri"}.get(domain, "general")


def _run_enhancement(
    *,
    hr: Image.Image,
    mode: str,
    method: str,
    alpha: float,
    tile_size: int,
    tile_overlap: int,
    config_path: str,
    checkpoint_path: str,
    domain: str,
) -> Dict[str, Any]:
    lr      = bicubic_downsample(hr, SCALE)
    bicubic = bicubic_upsample(lr, SCALE)
    esrgan: Optional[Image.Image] = None
    hybrid: Optional[Image.Image] = None
    use_tiling = False

    if method in (METHOD_ESRGAN, METHOD_HYBRID):
        service    = get_service(config_path, checkpoint_path)
        use_tiling = (mode == "full") or (max(lr.size) > tile_size)
        prog       = st.progress(0.0, text="Running ESRGAN…")

        def _cb(done: int, total: int) -> None:
            if total > 0:
                prog.progress(done / total, text=f"ESRGAN — tile {done}/{total}")

        try:
            esrgan = service.enhance_pil_image(
                lr, use_tiling=use_tiling,
                tile_size=tile_size, tile_overlap=tile_overlap,
                progress_callback=_cb,
            )
        finally:
            prog.empty()

    if method == METHOD_HYBRID and esrgan is not None:
        hybrid = create_hybrid_bicubic_esrgan(bicubic, esrgan, alpha=alpha)

    final = (
        bicubic if method == METHOD_BICUBIC
        else (esrgan  or bicubic) if method == METHOD_ESRGAN
        else (hybrid  or bicubic)
    )

    try:
        from app_services.metrics_util import compute_lpips as _lpips_fn
        _has_lpips = True
    except Exception:
        _has_lpips = False

    def _metric_row(img: Image.Image) -> Dict[str, float]:
        row: Dict[str, float] = {
            "psnr": compute_psnr(img, hr, crop=METRIC_CROP),
            "ssim": compute_ssim(img, hr, crop=METRIC_CROP),
        }
        if _has_lpips:
            lp = _lpips_fn(img, hr)
            if lp is not None:
                row["lpips"] = lp
        return row

    metrics: Dict[str, Dict[str, float]] = {"bicubic": _metric_row(bicubic)}
    if esrgan is not None:
        metrics["esrgan"] = _metric_row(esrgan)
    if hybrid is not None:
        metrics["hybrid"] = _metric_row(hybrid)

    bic_desc    = analyze_image(bicubic)
    final_desc  = analyze_image(final)
    deltas      = compare_analysis(bic_desc, final_desc)
    findings    = generate_findings_report(
        before=bic_desc, after=final_desc, deltas=deltas,
        mode=_findings_mode(domain),
        method_label=(
            "Hybrid output"         if method == METHOD_HYBRID
            else "ESRGAN output"    if method == METHOD_ESRGAN
            else "Bicubic baseline output"
        ),
    )

    # ── Change map ────────────────────────────────────────────────────────────
    change_map_payload: Optional[Dict[str, Any]] = None
    if method != METHOD_BICUBIC:
        cm   = compute_change_map(final_image=final, baseline_image=bicubic)
        bbox = cm.bbox

        from PIL import ImageDraw
        hr_bbox    = hr.copy()
        final_bbox = final.copy()
        if bbox is not None:
            x, y, w, h = bbox
            lw = max(2, int(round(min(final.size) * 0.005)))
            for target in (hr_bbox, final_bbox):
                ImageDraw.Draw(target).rectangle([x, y, x + w, y + h], outline=(79, 195, 184), width=lw)

        DISP = 640
        roi_bic_z = roi_fin_z = roi_hr_z = None
        if cm.roi_crop_baseline is not None and cm.roi_crop_final is not None:
            rw, rh   = cm.roi_crop_final.size
            roi_hr_c = hr.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])) if bbox else None
            zf       = DISP / max(rw, rh)
            if zf > 1.0:
                t          = (int(rw * zf), int(rh * zf))
                roi_bic_z  = cm.roi_crop_baseline.resize(t, Image.LANCZOS)
                roi_fin_z  = cm.roi_crop_final.resize(t, Image.LANCZOS)
                roi_hr_z   = roi_hr_c.resize(t, Image.LANCZOS) if roi_hr_c else None
            else:
                roi_bic_z = cm.roi_crop_baseline
                roi_fin_z = cm.roi_crop_final
                roi_hr_z  = roi_hr_c

        # Export-compatible side-by-side panel
        roi_panel = None
        if cm.roi_crop_baseline is not None and cm.roi_crop_final is not None:
            from app_services.visuals import _add_label_band
            l     = _add_label_band(cm.roi_crop_baseline, "Bicubic ROI")
            r     = _add_label_band(cm.roi_crop_final, f"{method} ROI")
            ph    = max(l.height, r.height)
            panel = Image.new("RGB", (l.width + r.width + 8, ph), (14, 17, 23))
            panel.paste(l, (0, 0))
            panel.paste(r, (l.width + 8, 0))
            roi_panel = panel

        change_map_payload = {
            "heatmap":          cm.heatmap_rgb,
            "overlay":          cm.overlay,
            "overlay_with_bbox": cm.overlay_with_bbox,
            "hr_with_bbox":     hr_bbox,
            "final_with_bbox":  final_bbox,
            "roi_baseline_zoom": roi_bic_z,
            "roi_final_zoom":   roi_fin_z,
            "roi_hr_zoom":      roi_hr_z,
            "roi_panel":        roi_panel,
            "metadata":         cm.metadata,
        }

    return {
        "hr": hr, "lr": lr, "bicubic": bicubic, "esrgan": esrgan,
        "hybrid": hybrid, "final": final, "method": method, "alpha": alpha,
        "metrics": metrics, "bicubic_desc": bic_desc, "final_desc": final_desc,
        "descriptor_deltas": deltas, "findings": findings,
        "change_map": change_map_payload, "used_tiling": use_tiling, "domain": domain,
    }


def _save_case(
    *,
    original: Image.Image,
    hr: Image.Image, lr: Image.Image,
    bicubic: Image.Image,
    esrgan: Optional[Image.Image],
    hybrid: Optional[Image.Image],
    final: Image.Image,
    method: str, alpha: float,
    crop_panel: Image.Image,
    metrics: Dict[str, Dict[str, float]],
    descriptors: Dict[str, Any],
    descriptor_deltas: Dict[str, Any],
    findings: Dict[str, Any],
    change_map: Optional[Dict[str, Any]],
    domain: str, input_filename: str, mode_label: str,
    used_tiling: bool,
    best_metrics: Dict[str, Any],
) -> Path:
    root = Path("ui_outputs")
    root.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    case_dir = root / f"case_{ts}"
    case_dir.mkdir(parents=True, exist_ok=True)

    original.save(case_dir / "original_input.png")
    hr.save(case_dir / "hr_aligned.png")
    lr.save(case_dir / "lr_synthetic.png")
    bicubic.save(case_dir / "bicubic_reference.png")
    final.save(case_dir / "final_output.png")
    crop_panel.save(case_dir / "side_by_side_crop.png")
    if esrgan is not None:
        esrgan.save(case_dir / "esrgan_output.png")
    if hybrid is not None:
        hybrid.save(case_dir / "hybrid_output.png")

    if change_map is not None:
        cm = change_map
        for name, key in [
            ("change_map_heatmap.png", "heatmap"),
            ("change_map_overlay.png", "overlay"),
            ("change_map_overlay_bbox.png", "overlay_with_bbox"),
            ("change_map_roi.png", "roi_panel"),
        ]:
            if cm.get(key) is not None:
                cm[key].save(case_dir / name)
        (case_dir / "change_map.json").write_text(
            json.dumps(cm.get("metadata", {}), indent=2), encoding="utf-8"
        )

    has_lpips = "lpips" in metrics.get("bicubic", {})
    if has_lpips:
        hdr = f"| Method | PSNR (dB, crop={METRIC_CROP}) | SSIM (crop={METRIC_CROP}) | LPIPS ↓ |"
        sep = "|---|---:|---:|---:|"
        def _r(n, m):
            return f"| {n} | {m['psnr']:.3f} | {m['ssim']:.4f} | {m.get('lpips', float('nan')):.4f} |"
    else:
        hdr = f"| Method | PSNR (dB, crop={METRIC_CROP}) | SSIM (crop={METRIC_CROP}) |"
        sep = "|---|---:|---:|"
        def _r(n, m):
            return f"| {n} | {m['psnr']:.3f} | {m['ssim']:.4f} |"

    tbl_rows = [hdr, sep, _r(f"Bicubic ×{SCALE}", metrics["bicubic"])]
    if "esrgan" in metrics:
        tbl_rows.append(_r("ESRGAN", metrics["esrgan"]))
    if "hybrid" in metrics:
        tbl_rows.append(_r(f"Hybrid (α={alpha:.2f})", metrics["hybrid"]))

    vp  = best_metrics.get("val_psnr")
    vs  = best_metrics.get("val_ssim")
    vps = f"{vp:.4f} dB" if isinstance(vp, (int, float)) else "—"
    vss = f"{vs:.4f}"    if isinstance(vs, (int, float)) else "—"

    cm_section = ""
    if change_map and change_map.get("metadata"):
        meta = change_map["metadata"]
        bb   = meta.get("bbox")
        bbs  = f"x={bb[0]}, y={bb[1]}, w={bb[2]}, h={bb[3]}" if isinstance(bb, list) and len(bb) == 4 else "—"
        cm_section = f"""
## Super-Resolution Change Map

- Mean |Δ|: {meta.get('mean_diff', 0):.4f}
- Max |Δ|: {meta.get('max_diff', 0):.4f}
- Top-percentile threshold: {meta.get('bbox_top_percentile', 95):.0f}%
- Smooth σ: {meta.get('smooth_sigma', 4)} px
- Bbox: {bbs}

Not a diagnostic region — indicates where SR changed pixel values most strongly.
"""

    report = f"""# Super-Resolution Enhancement Report

- **Domain:** {domain}
- **Method:** {method}
- **Alpha (ESRGAN strength):** {alpha:.2f}
- **Input filename:** `{input_filename}`
- **Timestamp:** {ts}
- **Inference mode:** {mode_label}
- **Tiled inference:** {"Yes" if used_tiling else "No"}
- **HR (aligned) size:** {hr.width}×{hr.height}
- **Output size:** {final.width}×{final.height}
- **Model:** ESRGAN ×{SCALE} — epoch {best_metrics.get('epoch', '—')}
- **Training val PSNR / SSIM:** {vps} / {vss}

## Metrics (vs. HR ground truth)

{chr(10).join(tbl_rows)}

## Assistive Review

{findings_to_markdown(findings)}
{cm_section}
## Image Descriptors (final output)

| Descriptor | Bicubic | Final | Δ |
|---|---:|---:|---:|
| Sharpness (Laplacian var) | {descriptors.get('sharpness', 0):.3f} | {descriptors.get('sharpness', 0):.3f} | {descriptor_deltas.get('sharpness_delta', 0):+.3f} |
| Contrast (std dev) | {descriptors.get('contrast', 0):.3f} | {descriptors.get('contrast', 0):.3f} | {descriptor_deltas.get('contrast_delta', 0):+.3f} |
| Edge density (Canny) | {descriptors.get('edge_density', 0):.4f} | {descriptors.get('edge_density', 0):.4f} | {descriptor_deltas.get('edge_density_delta', 0):+.4f} |
| Entropy | {descriptors.get('entropy', 0):.3f} | {descriptors.get('entropy', 0):.3f} | {descriptor_deltas.get('entropy_delta', 0):+.3f} |

---

*Assistive visual review only. Not a diagnostic tool.*
"""
    (case_dir / "report.md").write_text(report, encoding="utf-8")
    (case_dir / "metrics.json").write_text(json.dumps({
        "domain": domain, "method": method, "alpha": alpha,
        "metrics": metrics, "descriptors": descriptors,
        "deltas": descriptor_deltas,
        "input_size": [hr.width, hr.height],
        "output_size": [final.width, final.height],
        "inference_mode": mode_label, "used_tiling": used_tiling,
        **({"change_map": change_map.get("metadata", {})} if change_map else {}),
    }, indent=2), encoding="utf-8")
    return case_dir


# ── SHARED UI COMPONENTS ──────────────────────────────────────────────────────

def _checkpoint_card(best: Dict[str, Any], label: str) -> None:
    if not best:
        st.markdown(
            f"<div class='esr-card'>"
            f"<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.4rem;'>"
            f"<span class='esr-tag'>Model</span>"
            f"<span style='font-size:0.75rem;color:#F87171;'>● No checkpoint</span>"
            f"</div>"
            f"<div class='esr-kv'>{label}</div>"
            f"<div class='esr-caption' style='margin-top:0.25rem;'>No trained checkpoint found. "
            f"Train this domain first — see README.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        return
    epoch = best.get("epoch", "—")
    psnr  = best.get("val_psnr")
    ssim  = best.get("val_ssim")
    ps    = f"{psnr:.2f} dB" if isinstance(psnr, (int, float)) else "—"
    ss    = f"{ssim:.4f}"    if isinstance(ssim, (int, float)) else "—"
    sel   = best.get("selection", "max val PSNR")
    slbl  = {"best_post_gan_ssim": "max SSIM (post-GAN)", "best_ssim": "max SSIM"}.get(sel, "max PSNR")
    st.markdown(
        f"<div class='esr-card'>"
        f"<div style='display:flex; align-items:center; gap:0.65rem; margin-bottom:0.45rem;'>"
        f"<span class='esr-tag'>Best checkpoint</span>"
        f"<span class='esr-caption'>{label}</span>"
        f"</div>"
        f"<div style='display:flex; gap:1.4rem; flex-wrap:wrap;'>"
        f"<div><small>Epoch</small><div class='esr-kv'>{epoch}</div></div>"
        f"<div><small>Val PSNR</small><div class='esr-kv'>{ps}</div></div>"
        f"<div><small>Val SSIM</small><div class='esr-kv'>{ss}</div></div>"
        f"<div><small>Selected by</small><div class='esr-kv'>{slbl}</div></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )


def _placeholder(msg: str, hint: str = "") -> None:
    hint_html = (
        f"<div class='esr-placeholder-hint'>{hint}</div>" if hint else ""
    )
    st.markdown(
        f"<div class='esr-placeholder'>"
        f"<div class='esr-placeholder-msg'>{msg}</div>"
        f"{hint_html}</div>",
        unsafe_allow_html=True,
    )


def _warn_banner(body: str) -> None:
    st.markdown(
        f"<div class='esr-card-warn'>"
        f"<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.35rem;'>"
        f"<span class='esr-tag-warn'>Research only</span>"
        f"</div>"
        f"<div class='esr-caption'>{body}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── TAB: OVERVIEW ─────────────────────────────────────────────────────────────

def _tab_overview(
    upload_result: Optional[Tuple[Image.Image, str]],
    state: Dict[str, Any],
    best_metrics: Dict[str, Any],
    domain_label: str,
) -> None:
    if upload_result is None:
        col_start, col_how = st.columns([3, 2], gap="large")
        with col_start:
            st.markdown(
                f"<div style='font-family:Figtree,sans-serif;font-size:1rem;"
                f"font-weight:700;color:{TEXT};margin-bottom:0.6rem;letter-spacing:-0.01em;'>"
                f"Get started</div>",
                unsafe_allow_html=True,
            )
            steps = [
                "Select an image domain in the sidebar — BACH histopathology or Brain MRI.",
                "Upload a medical image (.png / .jpg / .tif) via the sidebar uploader.",
                "Choose an enhancement method: Hybrid, ESRGAN-only, or Bicubic baseline.",
                "Click <strong>Run enhancement</strong>, then explore results in the tabs.",
            ]
            st.markdown(
                f"<div style='background:{BG_CARD};border:1px solid {BORDER};"
                f"border-radius:10px;padding:0.75rem 1rem;margin-bottom:0.9rem;'>",
                unsafe_allow_html=True,
            )
            for i, s in enumerate(steps, 1):
                st.markdown(
                    f"<div class='esr-step'>"
                    f"<span class='esr-step-num'>{i}</span>"
                    f"<span>{s}</span></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
            _checkpoint_card(best_metrics, domain_label)

        with col_how:
            with st.expander("How the pipeline works", expanded=True):
                st.markdown(
                    f"""
**×{SCALE} super-resolution pipeline:**

```
Upload → HR (aligned to ×{SCALE})
HR → bicubic ÷4 → LR  (synthetic)
LR → bicubic ×4 → Bicubic  (baseline)
LR → ESRGAN  ×4 → ESRGAN   (learned)
Hybrid = bicubic + α·(ESRGAN − bicubic)
```

Metrics (PSNR / SSIM / LPIPS) compare each output against the original HR.
This matches the training evaluation exactly, making comparisons honest.
                    """.strip()
                )
        return

    image, fname = upload_result
    col_img, col_meta = st.columns([3, 2], gap="large")

    with col_img:
        st.image(
            resize_for_display(image, 800),
            caption=f"{fname}  ·  {image.width}×{image.height} px",
            use_container_width=True,
        )

    with col_meta:
        st.markdown("#### Image")
        st.markdown(
            f"<div class='esr-card'>"
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:0.5rem 1rem;'>"
            f"<div><small>Width</small><div class='esr-kv'>{image.width} px</div></div>"
            f"<div><small>Height</small><div class='esr-kv'>{image.height} px</div></div>"
            f"<div><small>Megapixels</small><div class='esr-kv'>{image.width * image.height / 1e6:.2f}</div></div>"
            f"<div><small>Colour mode</small><div class='esr-kv'>{image.mode}</div></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown("#### Model")
        _checkpoint_card(best_metrics, domain_label)

        if "final" not in state:
            st.info("Click **Run enhancement** in the sidebar to process this image.")
        else:
            m      = state["metrics"]
            method = state.get("method", METHOD_HYBRID)
            final  = state["final"]
            key    = "hybrid" if "hybrid" in m else ("esrgan" if "esrgan" in m else "bicubic")
            psnr   = m[key]["psnr"]
            ssim   = m[key]["ssim"]
            st.success(f"Enhancement complete — {method}")
            c1, c2 = st.columns(2)
            c1.metric("PSNR", f"{psnr:.2f} dB")
            c2.metric("SSIM", f"{ssim:.4f}")
            st.caption(
                f"{final.width}×{final.height} px  ·  "
                f"{state.get('mode_label', '')}  ·  "
                f"{'tiled' if state.get('used_tiling') else 'single-pass'}"
            )


# ── TAB: BEFORE / AFTER ───────────────────────────────────────────────────────

def _tab_before_after(state: Dict[str, Any]) -> None:
    if "final" not in state:
        _placeholder("No results yet", "Run an enhancement first using the sidebar.")
        return

    lr: Image.Image     = state["lr"]
    bicubic: Image.Image = state["bicubic"]
    final: Image.Image  = state["final"]
    hr: Image.Image     = state["hr"]
    method: str         = state.get("method", METHOD_HYBRID)
    alpha: float        = float(state.get("alpha", DEFAULT_ALPHA))

    lr_w, lr_h = lr.size
    max_crop   = max(64, min(lr_w, lr_h, 512))

    with st.expander("Crop controls", expanded=True):
        col_ctrl, col_info = st.columns([3, 2], gap="large")
        with col_ctrl:
            default_side = min(DEFAULT_CROP_LR, max_crop)
            if max_crop > 64:
                lr_crop_side = st.slider(
                    "LR crop side (px)", 64, max_crop, default_side, step=32,
                    help=f"SR / Bicubic crop on the HR side = {SCALE}× this value.",
                )
            else:
                lr_crop_side = 64
                st.caption(f"LR crop side: {lr_crop_side} px (image too small to adjust)")
            default_box = default_lr_crop_box(lr.size, lr_crop_side)
            max_x = max(0, lr_w - lr_crop_side)
            max_y = max(0, lr_h - lr_crop_side)
            cx = st.slider("X offset", 0, max_x, default_box[0], step=max(1, lr_crop_side // 8)) if max_x else 0
            cy = st.slider("Y offset", 0, max_y, default_box[1], step=max(1, lr_crop_side // 8)) if max_y else 0

        with col_info:
            final_label = (
                f"Hybrid (α={alpha:.2f})" if method == METHOD_HYBRID
                else "ESRGAN"             if method == METHOD_ESRGAN
                else f"Bicubic ×{SCALE}"
            )
            st.markdown(
                f"<div class='esr-caption'>"
                f"HR: {hr.width}×{hr.height} &nbsp;·&nbsp; LR: {lr.width}×{lr.height}<br/>"
                f"Output: {final.width}×{final.height}<br/>"
                f"Panel crop (HR side): {lr_crop_side * SCALE}×{lr_crop_side * SCALE} px<br/>"
                f"Method: <strong style='color:{ACCENT}'>{method}</strong>"
                + (f" · α = {alpha:.2f}" if method == METHOD_HYBRID else "")
                + "</div>",
                unsafe_allow_html=True,
            )

    crop_box = (cx, cy, cx + lr_crop_side, cy + lr_crop_side)
    hr_crop_box = scale_crop_box(crop_box, SCALE)
    st.markdown(f"**Drag to compare** — Bicubic ×{SCALE} vs {final_label}")
    image_comparison(
        img1=bicubic.crop(hr_crop_box),
        img2=final.crop(hr_crop_box),
        label1=f"Bicubic ×{SCALE}",
        label2=final_label,
        width=700,
        in_memory=True,
    )

    panel = make_triple_crop_panel(
        lr_image=lr, bicubic_image=bicubic, sr_image=final,
        lr_crop_box=crop_box, scale=SCALE,
        lr_label="LR input (×4 nearest)",
        bicubic_label=f"Bicubic ×{SCALE}",
        sr_label=final_label,
    )
    st.image(panel, caption="Native pixel ratio — LR (×4 nearest) · Bicubic ×4 · Enhanced",
             use_container_width=True)
    st.session_state["crop_panel"] = panel

    col_dl, _ = st.columns([2, 4])
    with col_dl:
        st.download_button(
            "Download final output (.png)",
            data=_png_bytes(final),
            file_name=f"{method.replace(' ', '_').lower()}_output.png",
            mime="image/png",
            use_container_width=True,
        )

    # Change-map teaser
    cm = state.get("change_map")
    if cm and method != METHOD_BICUBIC:
        st.markdown("---")
        st.markdown("**Enhancement focus region** — most-changed area (full analysis in **Change Map** tab)")
        col_orig, col_zoom = st.columns(2, gap="small")
        with col_orig:
            st.image(
                cm.get("hr_with_bbox") or hr,
                caption="Original — teal box marks the focus region",
                use_container_width=True,
            )
        with col_zoom:
            zoom = cm.get("roi_final_zoom") or cm.get("overlay")
            if zoom is not None:
                st.image(zoom, caption=f"{method} — zoomed focus region", use_container_width=True)

    if method == METHOD_HYBRID and state.get("esrgan") is not None:
        with st.expander("ESRGAN detail residual heatmap"):
            residual = compute_detail_residual(bicubic, state["esrgan"])
            from app_services.change_map import _apply_colormap
            heat = Image.fromarray(_apply_colormap(residual), mode="RGB")
            st.image(
                heat,
                caption="Magnitude of ESRGAN's texture above the bicubic baseline. Bright = stronger learned detail. Not diagnostic.",
                use_container_width=True,
            )


# ── TAB: METRICS ──────────────────────────────────────────────────────────────

def _tab_metrics(state: Dict[str, Any]) -> None:
    if "metrics" not in state:
        _placeholder("No metrics yet", "Run an enhancement first.")
        return

    m         = state["metrics"]
    alpha     = float(state.get("alpha", DEFAULT_ALPHA))
    has_lpips = "lpips" in m.get("bicubic", {})

    rows = [(f"Bicubic ×{SCALE}", m["bicubic"])]
    if "esrgan" in m:
        rows.append(("ESRGAN", m["esrgan"]))
    if "hybrid" in m:
        rows.append((f"Hybrid (α={alpha:.2f})", m["hybrid"]))

    col_tbl, col_note = st.columns([3, 2], gap="large")
    with col_tbl:
        st.markdown("**Reference-based metrics (vs. HR ground truth)**")
        table: Dict[str, list] = {
            "Method":                            [r[0] for r in rows],
            f"PSNR dB (crop={METRIC_CROP})":     [f"{r[1]['psnr']:.3f}" for r in rows],
            f"SSIM (crop={METRIC_CROP})":         [f"{r[1]['ssim']:.4f}" for r in rows],
        }
        if has_lpips:
            table["LPIPS ↓"] = [f"{r[1]['lpips']:.4f}" if "lpips" in r[1] else "—" for r in rows]
        st.dataframe(table, use_container_width=True, hide_index=True)

        if "esrgan" in m:
            dp = m["esrgan"]["psnr"] - m["bicubic"]["psnr"]
            ds = m["esrgan"]["ssim"] - m["bicubic"]["ssim"]
            winner = "ESRGAN" if dp > 0 else "Bicubic"
            note = (
                f"Δ PSNR (ESRGAN − Bicubic): {_delta_span(dp)} &nbsp;·&nbsp; "
                f"Δ SSIM: {_delta_span(ds, 4)} &nbsp;·&nbsp; "
                f"PSNR leader: <strong>{winner}</strong>"
            )
            if has_lpips and "lpips" in m["esrgan"] and "lpips" in m["bicubic"]:
                dl   = m["esrgan"]["lpips"] - m["bicubic"]["lpips"]
                lpw  = "ESRGAN" if dl < 0 else "Bicubic"
                note += (
                    f"<br/>Δ LPIPS: {_delta_span(dl, 4, lower_is_better=True)} &nbsp;·&nbsp; "
                    f"perceptually closer: <strong>{lpw}</strong>"
                )
            if "hybrid" in m:
                hp = m["hybrid"]["psnr"] - m["bicubic"]["psnr"]
                hs = m["hybrid"]["ssim"] - m["bicubic"]["ssim"]
                note += (
                    f"<br/>Δ PSNR (Hybrid − Bicubic): {_delta_span(hp)} &nbsp;·&nbsp; "
                    f"Δ SSIM: {_delta_span(hs, 4)}"
                )
            st.markdown(
                f"<div class='esr-caption' style='margin-top:0.5rem;'>{note}</div>",
                unsafe_allow_html=True,
            )

    with col_note:
        st.markdown(f"**Convention** — PSNR / SSIM on RGB [0, 1] with {METRIC_CROP}px border crop (matches training; removes edge artefacts at tile boundaries).")
        st.markdown(
            "**Honest reading** — ESRGAN is a perceptual GAN trained to maximise visual quality, "
            "not PSNR. On a single image it often scores lower than bicubic on PSNR while "
            "recovering more visible texture. The hybrid method blends ESRGAN learned detail "
            "onto the bicubic base, trading sharpness for numerical stability."
        )
        if has_lpips:
            st.markdown(
                "**LPIPS** (AlexNet, lower = more perceptually similar) typically favours "
                "GAN-trained methods over pixel-wise reconstruction even when PSNR is lower."
            )

    with st.expander("Non-reference image descriptors"):
        bic    = state["bicubic_desc"]
        fnl    = state["final_desc"]
        deltas = state["descriptor_deltas"]
        st.caption("Computed on each output independently — no HR reference required.")
        st.dataframe(
            {
                "Descriptor": [
                    "Laplacian variance (sharpness)",
                    "Std dev (contrast)",
                    "Canny edge density",
                    "Entropy",
                ],
                "Bicubic": [
                    bic.get("sharpness", 0), bic.get("contrast", 0),
                    bic.get("edge_density", 0), bic.get("entropy", 0),
                ],
                "Final": [
                    fnl.get("sharpness", 0), fnl.get("contrast", 0),
                    fnl.get("edge_density", 0), fnl.get("entropy", 0),
                ],
                "Δ (Final − Bicubic)": [
                    deltas.get("sharpness_delta", 0), deltas.get("contrast_delta", 0),
                    deltas.get("edge_density_delta", 0), deltas.get("entropy_delta", 0),
                ],
            },
            use_container_width=True,
            hide_index=True,
        )


# ── TAB: CHANGE MAP ───────────────────────────────────────────────────────────

def _tab_change_map(state: Dict[str, Any]) -> None:
    if "final" not in state:
        _placeholder("No results yet", "Run an enhancement first.")
        return

    method = state.get("method", METHOD_HYBRID)
    if method == METHOD_BICUBIC:
        _placeholder(
            "No change map for Bicubic-only",
            "Switch to ESRGAN or Hybrid in the sidebar to compute a change map.",
        )
        return

    cm = state.get("change_map")
    if not cm:
        st.warning("Change map was not computed for this run.")
        return

    _warn_banner(
        "This map shows <strong>where the model changed pixel values most strongly</strong> "
        "relative to the bicubic baseline. It characterises the SR output — "
        "it is not lesion detection or clinical localisation."
    )

    meta     = cm.get("metadata", {})
    bbox     = meta.get("bbox")
    bbox_str = (
        f"x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}"
        if isinstance(bbox, list) and len(bbox) == 4 else "—"
    )

    col_orig, col_zoom = st.columns(2, gap="small")
    with col_orig:
        st.markdown("**Original — enhancement focus region**")
        st.image(
            cm.get("hr_with_bbox") or state.get("hr") or state["final"],
            caption="Teal box = region changed most by SR",
            use_container_width=True,
        )
    with col_zoom:
        st.markdown(f"**{method} — zoomed focus region**")
        zoom = cm.get("roi_final_zoom") or cm.get("roi_panel") or cm["overlay"]
        st.image(zoom, caption="Enhanced output inside highlighted region", use_container_width=True)

    domain      = state.get("domain", DOMAIN_GENERAL)
    use_hr_ref  = (domain == DOMAIN_BACH) and cm.get("roi_hr_zoom") is not None
    left_zoom   = cm.get("roi_hr_zoom") if use_hr_ref else cm.get("roi_baseline_zoom")
    left_caption = "Original (HR reference)" if use_hr_ref else f"Bicubic ×{SCALE} (baseline)"

    if left_zoom is not None and cm.get("roi_final_zoom") is not None:
        st.markdown("---")
        lbl = "HR reference vs. enhanced" if use_hr_ref else "Before / after — same region"
        st.markdown(f"**{lbl}**")
        col_b, col_f = st.columns(2, gap="small")
        with col_b:
            st.image(left_zoom, caption=left_caption, use_container_width=True)
        with col_f:
            st.image(cm["roi_final_zoom"], caption=f"{method} output", use_container_width=True)

    st.markdown(
        f"<div class='esr-caption' style='margin-top:0.5rem;'>"
        f"Mean |Δ|: <strong>{meta.get('mean_diff', 0):.4f}</strong>"
        f" &nbsp;·&nbsp; Max |Δ|: <strong>{meta.get('max_diff', 0):.4f}</strong>"
        f" &nbsp;·&nbsp; Top-percentile: {meta.get('bbox_top_percentile', 95):.0f}%"
        f" &nbsp;·&nbsp; Smooth σ: {meta.get('smooth_sigma', 4)} px"
        f" &nbsp;·&nbsp; Bbox: {bbox_str}"
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Full-image heatmap & overlay (technical view)", expanded=False):
        col_h, col_o = st.columns(2, gap="small")
        with col_h:
            st.image(cm["heatmap"], caption="Normalised |final − bicubic| across the full image",
                     use_container_width=True)
        with col_o:
            st.image(
                cm.get("overlay_with_bbox") or cm["overlay"],
                caption="Heatmap blended on output — teal box = strongest changed region",
                use_container_width=True,
            )


# ── TAB: REVIEW ───────────────────────────────────────────────────────────────

def _tab_review(state: Dict[str, Any]) -> None:
    if "findings" not in state:
        _placeholder("No results yet", "Run an enhancement first.")
        return

    f = state["findings"]
    _warn_banner(f.get("disclaimer", "Assistive visual review only. Not a diagnostic tool."))

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("**Assistive impression**")
        st.markdown(
            f"<div class='esr-card'>"
            f"<div class='esr-kv' style='font-size:0.94rem;line-height:1.6;'>"
            f"{f.get('assistive_impression', '')}</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("**Limitations**")
        st.markdown(
            f"<div class='esr-card'>"
            f"<div class='esr-caption'>{f.get('limitations', '')}</div></div>",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown("**Notable visible findings**")
        items = f.get("possible_findings", []) or ["No meaningful change detected on measured indicators."]
        for i, item in enumerate(items, 1):
            st.markdown(
                f"<div class='esr-step'>"
                f"<span class='esr-step-num'>{i}</span>"
                f"<span class='esr-caption'>{item}</span></div>",
                unsafe_allow_html=True,
            )


# ── TAB: PREDICTION ───────────────────────────────────────────────────────────

def _tab_prediction(
    upload_result: Optional[Tuple[Image.Image, str]],
    domain: str,
    state: Dict[str, Any],
) -> None:
    _warn_banner(
        "Classifier trained on BACH histopathology or BRISC2025 brain MRI. "
        "Outputs are model predictions for <strong>academic demonstration only</strong> — "
        "not clinical diagnosis. Do not use for clinical decisions."
    )

    dataset_map  = {DOMAIN_BACH: "bach", DOMAIN_BRAIN: "brisc"}
    dataset_name = dataset_map.get(domain)

    if dataset_name is None:
        _placeholder(
            "Domain not supported for classification",
            "Select Histopathology / BACH or Brain MRI in the sidebar.",
        )
        return

    with st.spinner("Loading classifier…"):
        svc = get_classifier_service(dataset_name)

    if svc is None:
        st.warning(
            f"No trained classifier found for **{dataset_name.upper()}**. "
            f"Run `python scripts/train_classifiers.py --dataset {dataset_name} --arch resnet50` first."
        )
        return

    st.caption(f"Model: {svc.label}")

    if upload_result is None:
        _placeholder("Upload an image first", "Use the sidebar uploader, then return here.")
        return

    original_image, _ = upload_result
    esrgan_image      = state.get("esrgan")

    options = ["Original uploaded image"]
    if esrgan_image is not None:
        options.append("ESRGAN-enhanced image")

    choice = st.radio(
        "Image to classify:",
        options,
        horizontal=True,
        help="Comparing predictions on original vs ESRGAN-enhanced lets you live-demo Phase 3 findings.",
    )
    target = (
        esrgan_image
        if (choice == "ESRGAN-enhanced image" and esrgan_image is not None)
        else original_image
    )

    with st.spinner("Running inference…"):
        result = svc.predict(target)

    col_pred, col_chart = st.columns([1, 1], gap="large")
    with col_pred:
        st.markdown("**Predicted class**")
        st.markdown(
            f"<div class='esr-card'>"
            f"<div style='font-size:2rem;font-weight:800;color:{ACCENT};margin-bottom:0.25rem;'>"
            f"{result.predicted_class}</div>"
            f"<small>Confidence</small>"
            f"<div class='esr-kv'>{result.confidence * 100:.1f}%</div>"
            f"<div class='esr-caption' style='margin-top:0.4rem;'>"
            f"Softmax probability for the top class. "
            f"Values below ~50% indicate model uncertainty.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_chart:
        st.markdown("**Probability distribution**")
        fig, ax = plt.subplots(figsize=(5, max(2.5, len(result.class_names) * 0.55)))
        colors  = [ACCENT if c == result.predicted_class else "#2c3a50" for c in result.class_names]
        bars    = ax.barh(result.class_names, result.probabilities, color=colors, height=0.55)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability", color=CAPTION, fontsize=9)
        for bar, p in zip(bars, result.probabilities):
            ax.text(
                min(p + 0.02, 0.96),
                bar.get_y() + bar.get_height() / 2,
                f"{p:.3f}", va="center", ha="left", fontsize=9, color="#c8cdd6",
            )
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        ax.tick_params(colors=CAPTION, labelsize=9)
        ax.xaxis.label.set_color(CAPTION)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        fig.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("---")
    st.markdown("**Grad-CAM — spatial attention**")
    with st.spinner("Generating Grad-CAM…"):
        overlay = svc.gradcam_overlay(target)

    if overlay is not None:
        col_cam, col_inp = st.columns(2, gap="small")
        with col_inp:
            st.image(target.resize((224, 224)), caption="Input (224×224 centre crop)",
                     use_container_width=True)
        with col_cam:
            st.image(overlay, caption="Grad-CAM — red = highest attention for predicted class",
                     use_container_width=True)
        st.markdown(
            "<div class='esr-caption'>"
            "Grad-CAM highlights the spatial regions most influential for the prediction. "
            "For your thesis analysis, verify that activations correspond to biologically "
            "plausible regions."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "Grad-CAM requires `pytorch-grad-cam`. "
            "Install it with `pip install grad-cam>=1.4.8`."
        )


# ── TAB: EXPORT ───────────────────────────────────────────────────────────────

def _tab_export(
    state: Dict[str, Any],
    input_filename: str,
    mode_label: str,
    best_metrics: Dict[str, Any],
) -> None:
    if "final" not in state:
        _placeholder("No results to export", "Run an enhancement first.")
        return

    final  = state["final"]
    method = state.get("method", METHOD_HYBRID)

    col_save, col_dl = st.columns(2, gap="large")
    with col_save:
        st.markdown("**Save case folder**")
        st.markdown(
            "<div class='esr-caption'>"
            "Writes a timestamped folder to <code>ui_outputs/</code> containing all pipeline "
            "images, change-map artifacts, <code>report.md</code>, and <code>metrics.json</code>."
            "</div><br/>",
            unsafe_allow_html=True,
        )
        if st.button("Export to ui_outputs/", type="primary", use_container_width=True):
            with st.spinner("Exporting…"):
                case_dir = _save_case(
                    original=state["original"],
                    hr=state["hr"], lr=state["lr"],
                    bicubic=state["bicubic"],
                    esrgan=state.get("esrgan"), hybrid=state.get("hybrid"),
                    final=final,
                    method=method, alpha=float(state.get("alpha", DEFAULT_ALPHA)),
                    crop_panel=state.get("crop_panel", final),
                    metrics=state["metrics"],
                    descriptors=state["final_desc"],
                    descriptor_deltas=state["descriptor_deltas"],
                    findings=state["findings"],
                    change_map=state.get("change_map"),
                    domain=state.get("domain", DOMAIN_GENERAL),
                    input_filename=input_filename,
                    mode_label=mode_label,
                    used_tiling=bool(state.get("used_tiling", False)),
                    best_metrics=best_metrics,
                )
            st.success(f"Exported → `{case_dir}`")
            st.session_state["last_case_dir"] = str(case_dir)

        if st.session_state.get("last_case_dir"):
            st.caption(f"Last export: `{st.session_state['last_case_dir']}`")

    with col_dl:
        st.markdown("**Download outputs**")
        st.download_button(
            f"Final output — {method.lower()} (.png)",
            data=_png_bytes(final),
            file_name=f"{method.replace(' ', '_').lower()}_output.png",
            mime="image/png",
            use_container_width=True,
        )
        st.download_button(
            "Bicubic reference (.png)",
            data=_png_bytes(state["bicubic"]),
            file_name="bicubic_reference.png",
            mime="image/png",
            use_container_width=True,
        )
        if state.get("crop_panel") is not None:
            st.download_button(
                "Side-by-side crop panel (.png)",
                data=_png_bytes(state["crop_panel"]),
                file_name="crop_panel.png",
                mime="image/png",
                use_container_width=True,
            )
        if state.get("esrgan") is not None and method != METHOD_ESRGAN:
            st.download_button(
                "Raw ESRGAN output (.png)",
                data=_png_bytes(state["esrgan"]),
                file_name="esrgan_raw_output.png",
                mime="image/png",
                use_container_width=True,
            )


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Page header ──────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:1rem;flex-wrap:wrap;"
        f"padding:0.8rem 1rem;margin-bottom:0.5rem;"
        f"background:linear-gradient(135deg,{BG_CARD} 0%,{BG_CARD2} 100%);"
        f"border:1px solid {BORDER};border-radius:12px;"
        f"box-shadow:0 1px 4px rgba(0,0,0,0.3);'>"
        f"<div style='width:2.6rem;height:2.6rem;border-radius:10px;"
        f"background:linear-gradient(135deg,#0891B2,{ACCENT});"
        f"display:flex;align-items:center;justify-content:center;"
        f"font-family:Figtree,sans-serif;font-size:1.1rem;font-weight:800;color:#06151A;"
        f"flex-shrink:0;box-shadow:0 2px 8px rgba(79,195,184,0.3);'>SR</div>"
        f"<div style='flex:1;min-width:0;'>"
        f"<div style='display:flex;align-items:center;gap:0.65rem;flex-wrap:wrap;'>"
        f"<h1 style='margin:0;font-family:Figtree,sans-serif;font-size:1.35rem;"
        f"font-weight:800;letter-spacing:-0.025em;color:{TEXT};'>Stable-ESRGAN Medical Imaging</h1>"
        f"<span class='esr-tag'>BSc Thesis</span>"
        f"</div>"
        f"<div style='font-size:0.78rem;color:{CAPTION};margin-top:0.15rem;"
        f"font-family:Noto Sans,sans-serif;'>"
        f"{TAGLINE} &nbsp;·&nbsp; {AUTHOR_NAME} &nbsp;·&nbsp; {UNIVERSITY}"
        f"</div></div>"
        f"<div style='display:flex;gap:0.5rem;flex-shrink:0;'>"
        f"<div style='text-align:center;padding:0.3rem 0.7rem;"
        f"background:rgba(79,195,184,0.08);border:1px solid {BORDER2};border-radius:8px;'>"
        f"<div style='font-family:Figtree,sans-serif;font-size:1rem;font-weight:700;color:{ACCENT};'>×{SCALE}</div>"
        f"<div style='font-size:0.6rem;color:{CAPTION};text-transform:uppercase;letter-spacing:0.08em;'>Scale</div>"
        f"</div>"
        f"<div style='text-align:center;padding:0.3rem 0.7rem;"
        f"background:rgba(79,195,184,0.08);border:1px solid {BORDER2};border-radius:8px;'>"
        f"<div style='font-family:Figtree,sans-serif;font-size:1rem;font-weight:700;color:{ACCENT};'>3</div>"
        f"<div style='font-size:0.6rem;color:{CAPTION};text-transform:uppercase;letter-spacing:0.08em;'>Domains</div>"
        f"</div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f"<div style='padding:0.75rem 0.9rem;margin-bottom:1rem;"
            f"background:rgba(79,195,184,0.07);border:1px solid {BORDER};"
            f"border-left:3px solid {ACCENT};border-radius:10px;'>"
            f"<div style='font-family:Figtree,sans-serif;font-weight:700;"
            f"font-size:0.95rem;color:{TEXT};letter-spacing:-0.01em;'>Controls</div>"
            f"<div style='font-size:0.72rem;color:{CAPTION};margin-top:0.1rem;'>"
            f"Configure and run the SR pipeline</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Domain
        st.markdown(
            f"<div style='font-family:Figtree,sans-serif;font-size:0.72rem;"
            f"font-weight:700;text-transform:uppercase;letter-spacing:0.1em;"
            f"color:{CAPTION};margin-bottom:0.35rem;'>Image Domain</div>",
            unsafe_allow_html=True,
        )
        domain = st.radio(
            "Image domain",
            [DOMAIN_BACH, DOMAIN_BRAIN, DOMAIN_GENERAL],
            index=0,
            label_visibility="collapsed",
        )
        domain_entry = DOMAIN_REGISTRY.get(domain)
        st.caption("Choose the domain closest to your image for best results.")

        st.markdown("---")

        # Image source
        st.markdown(
            f"<div style='font-family:Figtree,sans-serif;font-size:0.72rem;"
            f"font-weight:700;text-transform:uppercase;letter-spacing:0.1em;"
            f"color:{CAPTION};margin-bottom:0.35rem;'>Image Source</div>",
            unsafe_allow_html=True,
        )
        sample_entry = SAMPLE_IMAGES.get(domain)
        source_options = ["Sample", "Upload my own"] if sample_entry else ["Upload my own"]
        source_choice = st.radio(
            "Image source",
            source_options,
            index=0,
            label_visibility="collapsed",
        )

        uploaded_file = None
        sample_image: Optional[Tuple[Image.Image, str]] = None
        if source_choice == "Sample" and sample_entry:
            st.caption(sample_entry["label"])
            try:
                sample_pil = Image.open(sample_entry["path"]).convert("RGB")
                sample_image = (sample_pil, Path(sample_entry["path"]).name)
            except Exception as e:
                st.error(f"Could not load curated sample: {e}")
        else:
            uploaded_file = st.file_uploader(
                "Medical image",
                type=["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"],
                label_visibility="collapsed",
                help="Treated as HR ground truth. LR is synthesised via bicubic ÷4, matching the training pipeline.",
            )

        st.markdown("---")

        # Method
        st.markdown(
            f"<div style='font-family:Figtree,sans-serif;font-size:0.72rem;"
            f"font-weight:700;text-transform:uppercase;letter-spacing:0.1em;"
            f"color:{CAPTION};margin-bottom:0.35rem;'>Enhancement Method</div>",
            unsafe_allow_html=True,
        )
        method = st.radio(
            "Method",
            [METHOD_ESRGAN, METHOD_HYBRID, METHOD_BICUBIC],
            index=0,
            label_visibility="collapsed",
            help=(
                "ESRGAN only = raw network output (matches reported benchmark numbers). "
                "Hybrid = bicubic + α·(ESRGAN−bicubic). "
                "Bicubic only = non-learned baseline."
            ),
        )

        if method == METHOD_HYBRID:
            with st.expander("α boost mode (allow > 1.0)", expanded=False):
                allow_boost = st.checkbox(
                    "Allow α > 1.0",
                    value=False,
                    help="Output pixels are still clamped to [0, 1].",
                )
            alpha_max = ALPHA_BOOST_MAX if allow_boost else 1.0
            alpha = st.slider(
                "ESRGAN strength (α)",
                0.0, alpha_max,
                min(DEFAULT_ALPHA, alpha_max),
                step=0.05,
                help="0 = pure bicubic  ·  1 = pure ESRGAN  ·  values between blend both.",
            )
        else:
            alpha = DEFAULT_ALPHA

        st.markdown("---")

        # Inference mode
        st.markdown(
            f"<div style='font-family:Figtree,sans-serif;font-size:0.72rem;"
            f"font-weight:700;text-transform:uppercase;letter-spacing:0.1em;"
            f"color:{CAPTION};margin-bottom:0.35rem;'>Inference Mode</div>",
            unsafe_allow_html=True,
        )
        mode_choice = st.radio(
            "Mode",
            ["Quick preview", "Full resolution"],
            index=0,
            label_visibility="collapsed",
            help=(
                f"Quick: resizes long side to ≤{QUICK_PREVIEW_LONG_SIDE}px (fast). "
                f"Full: tiled ESRGAN, capped at {MAX_HR_LONG_SIDE}px."
            ),
        )
        mode = "quick" if mode_choice.startswith("Quick") else "full"
        mode_label = (
            f"Quick preview (≤{QUICK_PREVIEW_LONG_SIDE}px)"
            if mode == "quick"
            else f"Full-resolution (tiled, ≤{MAX_HR_LONG_SIDE}px)"
        )

        with st.expander("Tiling parameters", expanded=False):
            tile_size    = st.slider("Tile size (px, LR space)",    128, 512, DEFAULT_TILE_SIZE,    step=64)
            tile_overlap = st.slider("Tile overlap (px, LR space)",   8,  64, DEFAULT_TILE_OVERLAP,  step=4)

        # Model paths
        if domain_entry is not None:
            default_ckpt, resolved_metrics_path = _resolve_checkpoint_pair(domain_entry)
            default_cfg  = domain_entry["config"]
        else:
            default_ckpt           = DEFAULT_CHECKPOINT
            resolved_metrics_path  = BEST_METRICS_PATH
            default_cfg            = DEFAULT_CONFIG

        locked = domain_entry is not None and domain != DOMAIN_GENERAL
        with st.expander("Model paths", expanded=(domain_entry is None)):
            config_path     = st.text_input("Config YAML",           value=default_cfg,   disabled=locked, key=f"cfg_{domain}")
            checkpoint_path = st.text_input("Generator checkpoint",  value=default_ckpt,  disabled=locked, key=f"ckpt_{domain}")
            if locked:
                st.caption("Paths are set automatically for the selected domain.")

        st.markdown("---")

        # Run button
        run_clicked = st.button(
            "Run enhancement",
            type="primary",
            use_container_width=True,
            disabled=(uploaded_file is None and sample_image is None),
        )
        if uploaded_file is None and sample_image is None:
            st.caption("Upload an image above to enable.")

    # ── Load state ────────────────────────────────────────────────────────────
    best_metrics = load_best_metrics(resolved_metrics_path)
    domain_label = domain_entry["label"] if domain_entry else f"ESRGAN ×{SCALE} (custom)"

    upload_result: Optional[Tuple[Image.Image, str]] = sample_image
    if upload_result is None and uploaded_file is not None:
        try:
            original      = Image.open(uploaded_file).convert("RGB")
            upload_result = (original, uploaded_file.name)
        except Exception as e:
            st.sidebar.error(f"Could not read file: {e}")

    state: Dict[str, Any] = st.session_state.get("result_state", {})
    input_filename         = st.session_state.get("input_filename", "(none)")

    # ── Run pipeline ─────────────────────────────────────────────────────────
    if run_clicked and upload_result is not None:
        original, fname = upload_result
        try:
            with st.spinner("Preparing image…"):
                hr, prep_note = _prepare_hr(original, mode)
            st.sidebar.caption(prep_note)
            result = _run_enhancement(
                hr=hr, mode=mode, method=method, alpha=alpha,
                tile_size=tile_size, tile_overlap=tile_overlap,
                config_path=config_path, checkpoint_path=checkpoint_path,
                domain=domain,
            )
            result["original"]   = original
            result["mode_label"] = mode_label
            st.session_state["result_state"]  = result
            st.session_state["input_filename"] = fname
            state          = result
            input_filename = fname
        except Exception as e:
            st.error(f"Enhancement failed: {e}")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    (
        tab_overview, tab_before, tab_metrics,
        tab_cm, tab_review, tab_predict, tab_export,
    ) = st.tabs([
        "Overview",
        "Before / After",
        "Metrics",
        "Change Map",
        "Review",
        "Prediction",
        "Export",
    ])

    with tab_overview:
        _tab_overview(upload_result, state, best_metrics, domain_label)

    with tab_before:
        _tab_before_after(state)

    with tab_metrics:
        _tab_metrics(state)

    with tab_cm:
        _tab_change_map(state)

    with tab_review:
        _tab_review(state)

    with tab_predict:
        _tab_prediction(upload_result, domain, state)

    with tab_export:
        _tab_export(state, input_filename, mode_label, best_metrics)


if __name__ == "__main__":
    main()
