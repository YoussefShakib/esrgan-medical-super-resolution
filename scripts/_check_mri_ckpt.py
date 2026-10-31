"""Check what epoch the MRI checkpoints were saved at."""
import sys
from pathlib import Path
import torch

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

device = torch.device("cpu")
for ckpt_name in ["best_generator.pth", "best_ssim_generator.pth"]:
    ckpt_path = root / "runs/esrgan_brain_mri_x4/checkpoints" / ckpt_name
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    keys = [k for k in ckpt.keys() if k != "model_state"]
    print(f"{ckpt_name}: epoch={ckpt.get('epoch', '?')}  extra_keys={keys}")
