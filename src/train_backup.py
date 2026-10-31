\
import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import SuperResolutionDataset
from losses import CharbonnierLoss, VGGFeatureExtractor
from metrics import batch_metrics
from models import build_model
from models.discriminator import Discriminator
from utils import CSVLogger, count_parameters, ensure_dir, get_device, load_config, save_json, set_seed


def save_comparison(input_tensor: torch.Tensor, sr: torch.Tensor, hr: torch.Tensor, path: Path) -> None:
    if input_tensor.shape[-2:] != hr.shape[-2:]:
        input_tensor = torch.nn.functional.interpolate(
            input_tensor, size=hr.shape[-2:], mode="bicubic", align_corners=False
        )
    grid = torch.cat([input_tensor[:1], sr[:1], hr[:1]], dim=0)
    save_image(grid, path, nrow=3)


def evaluate(model, loader, device, model_name):
    model.eval()
    metrics_list = []
    with torch.no_grad():
        for lr, hr in loader:
            lr = lr.to(device)
            hr = hr.to(device)
            sr = model(lr)
            metrics = batch_metrics(sr, hr)
            metrics_list.append(metrics)
    mean_psnr = sum(item["psnr"] for item in metrics_list) / len(metrics_list)
    mean_ssim = sum(item["ssim"] for item in metrics_list) / len(metrics_list)
    return {"psnr": mean_psnr, "ssim": mean_ssim}


def save_checkpoint(path: Path, model, optimizer, epoch, extra=None) -> None:
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if extra is not None:
        state.update(extra)
    torch.save(state, path)


def load_best_psnr(experiment_dir: Path) -> float:
    metrics_path = experiment_dir / "best_metrics.json"
    if not metrics_path.exists():
        return -1.0
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        return float(data.get("val_psnr", -1.0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return -1.0


def train_reconstruction(config):
    set_seed(config.get("seed", 42))
    device = get_device(config)

    experiment_dir = ensure_dir(Path(config["save_dir"]) / config["experiment_name"])
    checkpoints_dir = ensure_dir(experiment_dir / "checkpoints")
    samples_dir = ensure_dir(experiment_dir / "samples")
    shutil.copy2(config["config_path"], experiment_dir / "used_config.yaml")

    model_name = config["model_name"].lower()
    srcnn_mode = model_name == "srcnn"

    train_dataset = SuperResolutionDataset(
        hr_dir=config["dataset"]["train_hr_dir"],
        scale=config["scale"],
        patch_size=config["dataset"]["patch_size"],
        is_train=True,
        srcnn_mode=srcnn_mode,
        degradation_config=config["dataset"].get("degradation", {}),
        image_mode=config["dataset"].get("image_mode", "RGB"),
    )
    val_dataset = SuperResolutionDataset(
        hr_dir=config["dataset"]["val_hr_dir"],
        scale=config["scale"],
        patch_size=config["dataset"]["patch_size"],
        is_train=False,
        srcnn_mode=srcnn_mode,
        degradation_config={**config["dataset"].get("degradation", {}), "eval_crop_size": config["dataset"].get("eval_crop_size", 512)},
        image_mode=config["dataset"].get("image_mode", "RGB"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, config["training"]["num_workers"] // 2),
        pin_memory=True,
    )

    model = build_model(model_name, config["scale"]).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"], betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["training"].get("lr_step", 20),
        gamma=config["training"].get("lr_gamma", 0.5),
    )

    logger = CSVLogger(
        experiment_dir / "train_log.csv",
        ["epoch", "train_loss", "val_psnr", "val_ssim", "lr"],
    )

    best_psnr = load_best_psnr(experiment_dir)
    start_epoch = 1
    last_checkpoint_path = checkpoints_dir / "last.pth"
    if config["training"].get("resume", True) and last_checkpoint_path.exists():
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Resuming {model_name} from epoch {start_epoch}")

    print(f"Training {model_name} on {device} with {count_parameters(model):,} trainable parameters")

    if start_epoch > config["training"]["epochs"]:
        print("Training already completed for configured epochs. Increase training.epochs to continue.")
        return

    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['epochs']}")
        first_batch = None

        for lr, hr in progress_bar:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if first_batch is None:
                first_batch = (lr.detach().cpu(), sr.detach().cpu(), hr.detach().cpu())
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device, model_name)

        logger.write(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_psnr": val_metrics["psnr"],
                "val_ssim": val_metrics["ssim"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if first_batch is not None:
            save_comparison(*first_batch, path=samples_dir / f"epoch_{epoch:03d}.png")

        save_checkpoint(checkpoints_dir / "last.pth", model, optimizer, epoch)

        if val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            save_checkpoint(checkpoints_dir / "best.pth", model, optimizer, epoch)
            save_json(
                experiment_dir / "best_metrics.json",
                {
                    "epoch": epoch,
                    "val_psnr": val_metrics["psnr"],
                    "val_ssim": val_metrics["ssim"],
                    "model_name": model_name,
                },
            )

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_psnr={val_metrics['psnr']:.2f}, val_ssim={val_metrics['ssim']:.4f}"
        )


def d_hinge_loss(real_logits, fake_logits):
    real_loss = torch.mean(torch.relu(1.0 - real_logits))
    fake_loss = torch.mean(torch.relu(1.0 + fake_logits))
    return real_loss + fake_loss


def g_hinge_loss(fake_logits):
    return -torch.mean(fake_logits)


def train_esrgan(config):
    set_seed(config.get("seed", 42))
    device = get_device(config)

    experiment_dir = ensure_dir(Path(config["save_dir"]) / config["experiment_name"])
    checkpoints_dir = ensure_dir(experiment_dir / "checkpoints")
    samples_dir = ensure_dir(experiment_dir / "samples")
    shutil.copy2(config["config_path"], experiment_dir / "used_config.yaml")

    train_dataset = SuperResolutionDataset(
        hr_dir=config["dataset"]["train_hr_dir"],
        scale=config["scale"],
        patch_size=config["dataset"]["patch_size"],
        is_train=True,
        srcnn_mode=False,
        degradation_config=config["dataset"].get("degradation", {}),
        image_mode=config["dataset"].get("image_mode", "RGB"),
    )
    val_dataset = SuperResolutionDataset(
        hr_dir=config["dataset"]["val_hr_dir"],
        scale=config["scale"],
        patch_size=config["dataset"]["patch_size"],
        is_train=False,
        srcnn_mode=False,
        degradation_config={**config["dataset"].get("degradation", {}), "eval_crop_size": config["dataset"].get("eval_crop_size", 512)},
        image_mode=config["dataset"].get("image_mode", "RGB"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, config["training"]["num_workers"] // 2),
        pin_memory=True,
    )

    generator = build_model("esrgan", config["scale"]).to(device)
    discriminator = Discriminator().to(device)

    if config["training"].get("pretrained_generator"):
        checkpoint = torch.load(config["training"]["pretrained_generator"], map_location=device)
        generator.load_state_dict(checkpoint["model_state"], strict=False)
        print(f"Loaded pretrained generator from {config['training']['pretrained_generator']}")

    pixel_loss = CharbonnierLoss()
    perceptual_loss_net = VGGFeatureExtractor().to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=config["training"]["lr_g"], betas=(0.9, 0.99))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config["training"]["lr_d"], betas=(0.9, 0.99))

    logger = CSVLogger(
        experiment_dir / "train_log.csv",
        ["epoch", "g_total", "g_pixel", "g_perceptual", "g_adv", "d_loss", "val_psnr", "val_ssim", "lr_g", "lr_d"],
    )

    best_psnr = load_best_psnr(experiment_dir)
    start_epoch = 1
    last_generator_path = checkpoints_dir / "last_generator.pth"
    last_discriminator_path = checkpoints_dir / "last_discriminator.pth"
    if config["training"].get("resume", True) and last_generator_path.exists() and last_discriminator_path.exists():
        generator_checkpoint = torch.load(last_generator_path, map_location=device)
        discriminator_checkpoint = torch.load(last_discriminator_path, map_location=device)
        generator.load_state_dict(generator_checkpoint["model_state"])
        discriminator.load_state_dict(discriminator_checkpoint["model_state"])
        optimizer_g.load_state_dict(generator_checkpoint["optimizer_state"])
        optimizer_d.load_state_dict(discriminator_checkpoint["optimizer_state"])
        start_epoch = int(generator_checkpoint.get("epoch", 0)) + 1
        print(f"Resuming esrgan from epoch {start_epoch}")

    gan_start_epoch = config["training"].get("gan_start_epoch", 10)
    print(
        f"Training esrgan on {device} with {count_parameters(generator):,} generator params "
        f"and {count_parameters(discriminator):,} discriminator params"
    )

    if start_epoch > config["training"]["epochs"]:
        print("Training already completed for configured epochs. Increase training.epochs to continue.")
        return

    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        generator.train()
        discriminator.train()

        g_total_running = 0.0
        g_pixel_running = 0.0
        g_perc_running = 0.0
        g_adv_running = 0.0
        d_running = 0.0
        first_batch = None

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['epochs']}")
        for lr, hr in progress_bar:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            sr = generator(lr)

            pixel = pixel_loss(sr, hr) * config["training"]["pixel_weight"]

            sr_features = perceptual_loss_net(sr)
            hr_features = perceptual_loss_net(hr).detach()
            perceptual = torch.mean(torch.abs(sr_features - hr_features)) * config["training"]["perceptual_weight"]

            adv = torch.tensor(0.0, device=device)
            d_loss = torch.tensor(0.0, device=device)

            if epoch >= gan_start_epoch:
                optimizer_d.zero_grad()
                real_logits = discriminator(hr)
                fake_logits = discriminator(sr.detach())
                d_loss = d_hinge_loss(real_logits, fake_logits)
                d_loss.backward()
                optimizer_d.step()

                optimizer_g.zero_grad()
                fake_logits_g = discriminator(sr)
                adv = g_hinge_loss(fake_logits_g) * config["training"]["adv_weight"]
                total_g = pixel + perceptual + adv
                total_g.backward()
                optimizer_g.step()
            else:
                optimizer_g.zero_grad()
                total_g = pixel + perceptual
                total_g.backward()
                optimizer_g.step()

            g_total_running += float(total_g.item())
            g_pixel_running += float(pixel.item())
            g_perc_running += float(perceptual.item())
            g_adv_running += float(adv.item())
            d_running += float(d_loss.item())

            if first_batch is None:
                first_batch = (lr.detach().cpu(), sr.detach().cpu(), hr.detach().cpu())

            progress_bar.set_postfix(
                g=f"{total_g.item():.4f}",
                d=f"{d_loss.item():.4f}",
                phase="gan" if epoch >= gan_start_epoch else "warmup",
            )

        val_metrics = evaluate(generator, val_loader, device, "esrgan")

        logger.write(
            {
                "epoch": epoch,
                "g_total": g_total_running / len(train_loader),
                "g_pixel": g_pixel_running / len(train_loader),
                "g_perceptual": g_perc_running / len(train_loader),
                "g_adv": g_adv_running / len(train_loader),
                "d_loss": d_running / len(train_loader),
                "val_psnr": val_metrics["psnr"],
                "val_ssim": val_metrics["ssim"],
                "lr_g": optimizer_g.param_groups[0]["lr"],
                "lr_d": optimizer_d.param_groups[0]["lr"],
            }
        )

        if first_batch is not None:
            save_comparison(*first_batch, path=samples_dir / f"epoch_{epoch:03d}.png")

        torch.save(
            {
                "epoch": epoch,
                "model_state": generator.state_dict(),
                "optimizer_state": optimizer_g.state_dict(),
            },
            checkpoints_dir / "last_generator.pth",
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state": discriminator.state_dict(),
                "optimizer_state": optimizer_d.state_dict(),
            },
            checkpoints_dir / "last_discriminator.pth",
        )

        if val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": generator.state_dict(),
                    "optimizer_state": optimizer_g.state_dict(),
                },
                checkpoints_dir / "best_generator.pth",
            )
            save_json(
                experiment_dir / "best_metrics.json",
                {
                    "epoch": epoch,
                    "val_psnr": val_metrics["psnr"],
                    "val_ssim": val_metrics["ssim"],
                    "model_name": "esrgan",
                },
            )

        print(
            f"Epoch {epoch}: g_total={g_total_running / len(train_loader):.4f}, "
            f"d_loss={d_running / len(train_loader):.4f}, "
            f"val_psnr={val_metrics['psnr']:.2f}, val_ssim={val_metrics['ssim']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    config["config_path"] = args.config

    if config["model_name"].lower() == "esrgan":
        train_esrgan(config)
    else:
        train_reconstruction(config)


if __name__ == "__main__":
    main()
