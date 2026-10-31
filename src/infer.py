\
import argparse
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from models import build_model
from utils import load_config, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config")
    parser.add_argument("--checkpoint", required=True, help="Best or last checkpoint")
    parser.add_argument("--input", required=True, help="Input LR image path")
    parser.add_argument("--output", required=True, help="Output SR image path")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config)

    model = build_model(config["model_name"], config["scale"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    image = Image.open(args.input).convert("RGB")
    image_tensor = TF.to_tensor(image).unsqueeze(0).to(device)

    if config["model_name"].lower() == "srcnn":
        width, height = image.size
        image_tensor = torch.nn.functional.interpolate(
            image_tensor,
            size=(height * config["scale"], width * config["scale"]),
            mode="bicubic",
            align_corners=False,
        )

    with torch.no_grad():
        sr = model(image_tensor).clamp(0, 1)

    sr_image = TF.to_pil_image(sr.squeeze(0).cpu())
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sr_image.save(output_path)
    print(f"Saved result to {output_path}")


if __name__ == "__main__":
    main()
