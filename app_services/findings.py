"""Assistive-review report generator.

Produces a fixed 4-part structure:
    1. Assistive impression  (neutral 1-2 sentences)
    2. Notable visible findings  (bullets derived only from image stats)
    3. Limitations               (one sentence)
    4. Disclaimer                (one sentence)

All "findings" are derived from measurable deltas between the selected
enhancement output and
the bicubic baseline (both at the same upscaled resolution): change in Laplacian
energy, change in local variance, change in high-frequency content,
change in edge density. No classifier output. No clinical claims.
"""

from __future__ import annotations

from typing import Any, Dict, List


DISCLAIMER = "Assistive visual review only. Not a diagnostic tool."

LIMITATIONS_BY_MODE = {
    "histopathology": (
        "Use the checkpoint trained on the closest image domain. Super-resolution "
        "may sharpen visible tissue texture, but it can also hallucinate plausible "
        "fine detail that is not present in the original sample."
    ),
    "brain_mri": (
        "Use the checkpoint trained on the closest MRI-style image domain. "
        "Super-resolution may strengthen edge contrast and texture, but it can "
        "also hallucinate plausible detail that is not present in the original scan."
    ),
    "general": (
        "Best results usually come from a checkpoint trained on the closest image "
        "domain. Super-resolution may improve visual clarity while also introducing "
        "plausible synthetic detail."
    ),
}


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}"


def _notable_findings(
    bicubic: Dict[str, Any],
    esrgan: Dict[str, Any],
    deltas: Dict[str, Any],
) -> List[str]:
    """Turn raw deltas into neutral bullet observations."""
    bullets: List[str] = []

    sharp_delta = float(deltas.get("sharpness_delta", 0.0))
    sharp_base = float(bicubic.get("sharpness", 0.0)) or 1e-6
    sharp_pct = 100.0 * sharp_delta / sharp_base
    if abs(sharp_pct) >= 2.0:
        direction = "higher" if sharp_delta > 0 else "lower"
        bullets.append(
            f"Laplacian energy (local-edge sharpness) {direction} by {_fmt(sharp_pct, 1)}% "
            f"vs bicubic baseline at the same resolution."
        )
    else:
        bullets.append(
            "Laplacian energy (local-edge sharpness) is effectively unchanged vs bicubic baseline."
        )

    edge_delta = float(deltas.get("edge_density_delta", 0.0))
    if abs(edge_delta) >= 0.002:
        direction = "increased" if edge_delta > 0 else "decreased"
        bullets.append(
            f"Canny edge density {direction} by {_fmt(edge_delta, 4)} "
            f"(scale: [0, 1]; indicates change in high-frequency structure)."
        )
    else:
        bullets.append(
            "Canny edge density is effectively unchanged, suggesting no meaningful "
            "high-frequency restructuring."
        )

    contrast_delta = float(deltas.get("contrast_delta", 0.0))
    if abs(contrast_delta) >= 1.0:
        direction = "higher" if contrast_delta > 0 else "lower"
        bullets.append(
            f"Local pixel-intensity variance {direction} by {_fmt(contrast_delta, 2)} "
            f"(grayscale std dev in 0-255)."
        )

    return bullets


def _impression(deltas: Dict[str, Any], subject: str) -> str:
    """Neutral 1-2 sentence impression, honest about small effects."""
    sharp_delta = float(deltas.get("sharpness_delta", 0.0))
    edge_delta = float(deltas.get("edge_density_delta", 0.0))

    if sharp_delta > 0 and edge_delta > 0:
        return (
            f"{subject} shows increased local-edge sharpness and higher edge "
            "density than the bicubic baseline at the same resolution."
        )
    if sharp_delta > 0 or edge_delta > 0:
        return (
            f"{subject} shows a modest increase in at least one sharpness indicator "
            "relative to the bicubic baseline."
        )
    return (
        f"{subject} is numerically close to the bicubic baseline on the "
        "measured sharpness indicators; differences are small."
    )


def _limitations(mode: str) -> str:
    return LIMITATIONS_BY_MODE.get(mode, LIMITATIONS_BY_MODE["general"])


def generate_findings_report(
    before: Dict[str, Any],
    after: Dict[str, Any],
    deltas: Dict[str, Any],
    mode: str = "histopathology",
    method_label: str = "Selected enhancement output",
    classifier_result: Any = None,  # kept for signature compatibility; unused
) -> Dict[str, Any]:
    """Build the 4-part assistive review dict consumed by the UI and export."""
    subject = method_label.strip() if method_label and method_label.strip() else "Selected enhancement output"
    report = {
        "headline": "Assistive Visual Review",
        "assistive_impression": _impression(deltas, subject=subject),
        "possible_findings": _notable_findings(before, after, deltas),
        "limitations": _limitations(mode),
        "disclaimer": DISCLAIMER,
        "mode": mode,
        "method_label": subject,
    }
    # Back-compat key still used in a few downstream code paths:
    report["technical_note"] = (
        "Indicators are computed on the selected enhancement output and the "
        "bicubic-upsampled baseline at matched resolution. They are image-quality "
        "descriptors, not diagnostic biomarkers."
    )
    return report


def findings_to_markdown(findings: Dict[str, Any]) -> str:
    """Render the findings dict as a plain-markdown block for the report.md export."""
    lines: List[str] = []
    lines.append(f"### {findings.get('headline', 'Assistive Visual Review')}")
    lines.append("")
    lines.append("**1. Assistive impression**")
    lines.append("")
    lines.append(findings.get("assistive_impression", ""))
    lines.append("")
    lines.append("**2. Notable visible findings**")
    lines.append("")
    for bullet in findings.get("possible_findings", []) or ["(no meaningful change detected)"]:
        lines.append(f"- {bullet}")
    lines.append("")
    lines.append("**3. Limitations**")
    lines.append("")
    lines.append(findings.get("limitations", _limitations(findings.get("mode", "general"))))
    lines.append("")
    lines.append("**4. Disclaimer**")
    lines.append("")
    lines.append(findings.get("disclaimer", DISCLAIMER))
    return "\n".join(lines)
