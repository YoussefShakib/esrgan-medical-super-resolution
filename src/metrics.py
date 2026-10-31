from typing import Dict, Optional

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = tensor.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return array.astype(np.float32)


def calculate_psnr(sr: torch.Tensor, hr: torch.Tensor) -> float:
    sr_np = tensor_to_image(sr)
    hr_np = tensor_to_image(hr)
    return float(peak_signal_noise_ratio(hr_np, sr_np, data_range=1.0))


def calculate_ssim(sr: torch.Tensor, hr: torch.Tensor) -> float:
    sr_np = tensor_to_image(sr)
    hr_np = tensor_to_image(hr)
    return float(structural_similarity(hr_np, sr_np, channel_axis=2, data_range=1.0))


# Module-level singleton so the AlexNet backbone is only loaded once per process.
_lpips_net: Optional[object] = None


def _get_lpips_net(device: torch.device):
    global _lpips_net
    if _lpips_net is None:
        try:
            import lpips as _lpips_lib
            _lpips_net = _lpips_lib.LPIPS(net="alex", verbose=False).to(device)
        except ImportError as exc:
            raise ImportError(
                "lpips is required for LPIPS evaluation. "
                "Install it with: pip install lpips"
            ) from exc
    return _lpips_net


def calculate_lpips(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """Compute LPIPS distance (AlexNet, lower = more similar).

    Inputs: single-image tensors with shape (1,C,H,W) or (C,H,W),
    float32 in [0, 1]. No border crop is applied to match thesis convention.
    """
    device = sr.device
    net = _get_lpips_net(device)

    def _prep(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 3:
            t = t.unsqueeze(0)
        return t.clamp(0, 1).to(device) * 2.0 - 1.0  # scale to [-1, 1] as lpips expects

    with torch.no_grad():
        score = net(_prep(sr), _prep(hr))
    return float(score.mean().item())


def batch_metrics(sr: torch.Tensor, hr: torch.Tensor, compute_lpips: bool = False) -> Dict[str, float]:
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    for i in range(sr.size(0)):
        psnr_scores.append(calculate_psnr(sr[i:i+1], hr[i:i+1]))
        ssim_scores.append(calculate_ssim(sr[i:i+1], hr[i:i+1]))
        if compute_lpips:
            lpips_scores.append(calculate_lpips(sr[i:i+1], hr[i:i+1]))

    result: Dict[str, float] = {
        "psnr": float(np.mean(psnr_scores)),
        "ssim": float(np.mean(ssim_scores)),
    }
    if compute_lpips:
        result["lpips"] = float(np.mean(lpips_scores))
    return result
