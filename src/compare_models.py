import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from dataset import make_lr_from_hr
from metrics import batch_metrics
from models import build_model
from utils import get_device, load_config


def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    config = load_config(config_path)
    model = build_model(config['model_name'], config['scale']).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'], strict=False)
    model.eval()
    return config, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr-image', required=True)
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--labels', nargs='*')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--degradation', default='bicubic', choices=['bicubic', 'bilinear', 'nearest', 'lanczos'])
    args = parser.parse_args()

    if len(args.configs) != len(args.checkpoints):
        raise ValueError('configs and checkpoints must have the same length')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hr_image = Image.open(args.hr_image).convert('RGB')
    w, h = hr_image.size
    w = w - (w % args.scale)
    h = h - (h % args.scale)
    hr_image = hr_image.crop((0, 0, w, h))

    lr_image = make_lr_from_hr(hr_image, args.scale, degradation_config={'method': args.degradation}, is_train=False)
    hr_tensor = TF.to_tensor(hr_image).unsqueeze(0)
    lr_tensor = TF.to_tensor(lr_image).unsqueeze(0)

    bicubic_tensor = torch.nn.functional.interpolate(
        lr_tensor,
        size=(hr_tensor.shape[-2], hr_tensor.shape[-1]),
        mode='bicubic',
        align_corners=False,
    ).clamp(0, 1)

    results = [('Bicubic', bicubic_tensor, batch_metrics(bicubic_tensor, hr_tensor))]

    labels = args.labels or [Path(cfg).stem for cfg in args.configs]
    device = get_device({'device': 'auto'})

    for label, config_path, checkpoint_path in zip(labels, args.configs, args.checkpoints):
        config, model = load_model(config_path, checkpoint_path, device)
        input_tensor = lr_tensor.to(device)
        if config['model_name'].lower() == 'srcnn':
            input_tensor = torch.nn.functional.interpolate(
                input_tensor,
                size=(hr_tensor.shape[-2], hr_tensor.shape[-1]),
                mode='bicubic',
                align_corners=False,
            )
        with torch.no_grad():
            sr_tensor = model(input_tensor).clamp(0, 1).cpu()
        results.append((label, sr_tensor, batch_metrics(sr_tensor, hr_tensor)))

    columns = 2 + len(results)
    plt.figure(figsize=(4 * columns, 4))

    plt.subplot(1, columns, 1)
    plt.imshow(hr_image)
    plt.title('HR Ground Truth')
    plt.axis('off')

    plt.subplot(1, columns, 2)
    plt.imshow(lr_image)
    plt.title(f'LR x{args.scale}')
    plt.axis('off')

    for idx, (label, tensor, metrics) in enumerate(results, start=3):
        plt.subplot(1, columns, idx)
        plt.imshow(TF.to_pil_image(tensor.squeeze(0)))
        plt.title(f"{label}\nPSNR {metrics['psnr']:.2f} | SSIM {metrics['ssim']:.3f}")
        plt.axis('off')

    plt.tight_layout()
    figure_path = output_dir / 'comparison_panel.png'
    plt.savefig(figure_path, dpi=200, bbox_inches='tight')
    plt.close()

    metrics_path = output_dir / 'comparison_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({label: metrics for label, _, metrics in results}, f, indent=2)

    print(f'Saved panel to {figure_path}')
    print(f'Saved metrics to {metrics_path}')


if __name__ == '__main__':
    main()
