import argparse
from pathlib import Path

from PIL import Image

from dataset import list_images, make_lr_from_hr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr-dir', required=True)
    parser.add_argument('--lr-dir', required=True)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--method', default='bicubic', choices=['bicubic','bilinear','nearest','lanczos'])
    args = parser.parse_args()

    hr_dir = Path(args.hr_dir)
    lr_dir = Path(args.lr_dir)
    lr_dir.mkdir(parents=True, exist_ok=True)

    degradation = {'method': args.method}

    for img_path in list_images(hr_dir):
        img = Image.open(img_path).convert('RGB')
        lr = make_lr_from_hr(img, args.scale, degradation_config=degradation, is_train=False)
        lr.save(lr_dir / img_path.name)

    print(f'Saved LR dataset to {lr_dir}')


if __name__ == '__main__':
    main()
