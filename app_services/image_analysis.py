"""Image analysis and quality metrics."""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    """Convert PIL image to RGB numpy array."""
    return np.array(image.convert("RGB"))


def rgb_to_gray(image_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB array to grayscale."""
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


def calculate_sharpness(gray: np.ndarray) -> float:
    """Calculate image sharpness using Laplacian variance."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_contrast(gray: np.ndarray) -> float:
    """Calculate image contrast using standard deviation."""
    return float(gray.std())


def calculate_brightness(gray: np.ndarray) -> float:
    """Calculate image brightness using mean intensity."""
    return float(gray.mean())


def calculate_entropy(gray: np.ndarray) -> float:
    """Calculate image entropy (measure of detail/randomness)."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-12)
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum())


def calculate_edge_density(gray: np.ndarray) -> float:
    """Calculate edge density using Canny edge detection."""
    edges = cv2.Canny(gray, 100, 200)
    return float(edges.mean() / 255.0)


def calculate_channel_means(image_rgb: np.ndarray) -> Dict[str, float]:
    """Calculate mean intensity for each RGB channel."""
    return {
        "red_mean": float(image_rgb[:, :, 0].mean()),
        "green_mean": float(image_rgb[:, :, 1].mean()),
        "blue_mean": float(image_rgb[:, :, 2].mean()),
    }


def analyze_image(image: Image.Image) -> Dict[str, Any]:
    """
    Analyze image quality metrics.
    
    Args:
        image: PIL Image
        
    Returns:
        Dictionary with analysis metrics
        
    Raises:
        ValueError: If image is invalid
        RuntimeError: If analysis fails
    """
    try:
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(image)}")
        
        image_rgb = pil_to_rgb_array(image)
        gray = rgb_to_gray(image_rgb)

        width, height = image.size
        megapixels = (width * height) / 1_000_000

        result = {
            "width": width,
            "height": height,
            "megapixels": round(megapixels, 4),
            "mode": image.mode,
            "sharpness": round(calculate_sharpness(gray), 4),
            "contrast": round(calculate_contrast(gray), 4),
            "brightness": round(calculate_brightness(gray), 4),
            "entropy": round(calculate_entropy(gray), 4),
            "edge_density": round(calculate_edge_density(gray), 6),
        }
        result.update({k: round(v, 4) for k, v in calculate_channel_means(image_rgb).items()})
        return result
        
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Image analysis failed: {str(e)}") from e


def compare_analysis(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two analysis results to get deltas.
    
    Args:
        before: Analysis dict before enhancement
        after: Analysis dict after enhancement
        
    Returns:
        Dictionary with delta values
    """
    keys = [
        "sharpness",
        "contrast",
        "brightness",
        "entropy",
        "edge_density",
        "red_mean",
        "green_mean",
        "blue_mean",
    ]

    deltas: Dict[str, Any] = {}
    for key in keys:
        if key in before and key in after:
            deltas[f"{key}_delta"] = round(after[key] - before[key], 4)
    return deltas


def summarize_analysis(before: Dict[str, Any], after: Dict[str, Any], deltas: Dict[str, Any]) -> str:
    """
    Generate human-readable summary of analysis comparison.
    
    Args:
        before: Before analysis
        after: After analysis
        deltas: Delta values
        
    Returns:
        Summary string
    """
    points = []

    sharpness_delta = deltas.get("sharpness_delta", 0)
    edge_delta = deltas.get("edge_density_delta", 0)
    entropy_delta = deltas.get("entropy_delta", 0)
    contrast_delta = deltas.get("contrast_delta", 0)

    if sharpness_delta > 0:
        points.append("The enhanced image appears sharper than the input.")
    else:
        points.append("The enhanced image does not show a clear sharpness increase.")

    if edge_delta > 0:
        points.append("Edge density increased, which suggests stronger local boundary visibility.")
    else:
        points.append("Edge density did not increase noticeably after enhancement.")

    if contrast_delta > 0:
        points.append("Global contrast increased slightly after enhancement.")
    else:
        points.append("Global contrast stayed similar or decreased slightly after enhancement.")

    if entropy_delta > 0:
        points.append("Image entropy increased slightly, which may indicate richer visible detail.")
    else:
        points.append("Image entropy stayed similar or decreased slightly after enhancement.")

    return " ".join(points)


class ImageAnalysis:
    """Comprehensive image analysis and comparison."""
    
    def __init__(self, image: Image.Image):
        """
        Initialize with an image.
        
        Args:
            image: PIL Image to analyze
            
        Raises:
            ValueError: If image is invalid
        """
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(image)}")
        
        self.image = image
        self.analysis = None
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the image.
        
        Returns:
            Analysis results
        """
        self.analysis = analyze_image(self.image)
        return self.analysis
    
    def compare_with(self, other_image: Image.Image) -> Dict[str, Any]:
        """
        Compare with another image.
        
        Args:
            other_image: PIL Image to compare with
            
        Returns:
            Comparison deltas
            
        Raises:
            RuntimeError: If comparison fails
        """
        try:
            if self.analysis is None:
                self.analyze()
            
            other_analysis = analyze_image(other_image)
            comparison = {
                "before": self.analysis,
                "after": other_analysis,
                "deltas": compare_analysis(self.analysis, other_analysis),
                "summary": summarize_analysis(self.analysis, other_analysis, 
                                            compare_analysis(self.analysis, other_analysis)),
            }
            return comparison
        except Exception as e:
            raise RuntimeError(f"Comparison failed: {str(e)}") from e