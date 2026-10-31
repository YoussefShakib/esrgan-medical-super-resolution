"""Training script for super-resolution models with comprehensive improvements."""

import argparse
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.checkpoint import CheckpointManager
from core.config import load_config
from core.device import DeviceManager
from core.logging_setup import get_logger, setup_logging
from core.exceptions import SuperResolutionException, CheckpointError
from dataset import SuperResolutionDataset
from losses import CharbonnierLoss, VGGFeatureExtractor
from metrics import batch_metrics
from models import build_model
from models.discriminator import Discriminator
from utils import CSVLogger, count_parameters, ensure_dir, set_seed, save_json

logger = None  # Initialized in main


def _pin_memory_enabled(config, device: torch.device) -> bool:
    configured = getattr(config.training, "pin_memory", None)
    if configured is None:
        return device.type == "cuda"
    return bool(configured) and device.type == "cuda"


def _persistent_workers_enabled(config) -> bool:
    return bool(getattr(config.training, "persistent_workers", False)) and config.training.num_workers > 0


def _limit_validation_dataset(dataset, max_val_images: int):
    if max_val_images <= 0 or len(dataset) <= max_val_images:
        return dataset
    return Subset(dataset, list(range(max_val_images)))


def _should_run_periodic(epoch: int, every_n: int, final_epoch: int) -> bool:
    return epoch % max(1, every_n) == 0 or epoch == final_epoch


def save_comparison(input_tensor: torch.Tensor, sr: torch.Tensor, hr: torch.Tensor, path: Path) -> None:
    """Save comparison grid of LR, SR, and HR images."""
    if input_tensor.shape[-2:] != hr.shape[-2:]:
        input_tensor = torch.nn.functional.interpolate(
            input_tensor, size=hr.shape[-2:], mode="bicubic", align_corners=False
        )
    grid = torch.cat([input_tensor[:1], sr[:1], hr[:1]], dim=0)
    save_image(grid, path, nrow=3)
    logger.debug(f"Saved comparison: {path}")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
    compute_lpips: bool = False,
) -> dict:
    """Evaluate model on validation set, returning PSNR, SSIM and optionally LPIPS."""
    model.eval()
    metrics_list = []

    try:
        with torch.no_grad():
            for lr, hr in loader:
                lr = lr.to(device, non_blocking=True)
                hr = hr.to(device, non_blocking=True)
                sr = model(lr)
                metrics = batch_metrics(sr, hr, compute_lpips=compute_lpips)
                metrics_list.append(metrics)

        if not metrics_list:
            raise RuntimeError("No metrics computed during evaluation")

        mean_psnr = sum(item["psnr"] for item in metrics_list) / len(metrics_list)
        mean_ssim = sum(item["ssim"] for item in metrics_list) / len(metrics_list)

        if not (torch.isfinite(torch.tensor([mean_psnr, mean_ssim])).all().item()):
            logger.warning("NaN detected in metrics!")

        result = {"psnr": mean_psnr, "ssim": mean_ssim}
        if compute_lpips and all("lpips" in m for m in metrics_list):
            result["lpips"] = sum(item["lpips"] for item in metrics_list) / len(metrics_list)
        return result

    except Exception as e:
        raise RuntimeError(f"Evaluation failed: {str(e)}") from e


def train_reconstruction(config, compute_lpips: bool = False):
    """Train a reconstruction model (SRCNN, SRResNet, EDSR)."""
    global logger
    
    try:
        logger.info(f"Starting training for {config.model_name}")
        set_seed(config.seed)
        device = DeviceManager.setup(config.model_dump())

        experiment_dir = ensure_dir(Path(config.save_dir) / config.experiment_name)
        checkpoints_dir = ensure_dir(experiment_dir / "checkpoints")
        samples_dir = ensure_dir(experiment_dir / "samples")
        logs_dir = ensure_dir(experiment_dir / "logs")
        
        # Setup logging for this run
        run_logger = setup_logging(logs_dir, name=f"sr_project_{config.experiment_name}")
        
        # Copy config
        shutil.copy2(config.config_path, experiment_dir / "used_config.yaml")
        logger.info(f"Experiment dir: {experiment_dir}")

        model_name = config.model_name.lower()
        srcnn_mode = model_name == "srcnn"

        # Create datasets
        train_dataset = SuperResolutionDataset(
            hr_dir=config.dataset.train_hr_dir,
            scale=config.scale,
            patch_size=config.dataset.patch_size,
            is_train=True,
            srcnn_mode=srcnn_mode,
            degradation_config=config.dataset.degradation,
            image_mode=config.dataset.image_mode,
        )
        val_dataset = SuperResolutionDataset(
            hr_dir=config.dataset.val_hr_dir,
            scale=config.scale,
            patch_size=config.dataset.patch_size,
            is_train=False,
            srcnn_mode=srcnn_mode,
            degradation_config={**config.dataset.degradation, "eval_crop_size": config.dataset.eval_crop_size},
            image_mode=config.dataset.image_mode,
        )
        original_val_len = len(val_dataset)
        val_dataset = _limit_validation_dataset(val_dataset, int(config.training.max_val_images or 0))
        pin_memory = _pin_memory_enabled(config, device)
        persistent_workers = _persistent_workers_enabled(config)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=max(0, config.training.num_workers // 2),
            pin_memory=pin_memory,
            persistent_workers=bool(persistent_workers and max(0, config.training.num_workers // 2) > 0),
        )

        logger.info(f"Train set: {len(train_dataset)} images, Val set: {len(val_dataset)} images")
        if len(val_dataset) != original_val_len:
            logger.info(f"Validation subset enabled: {len(val_dataset)}/{original_val_len} images")

        # Build model
        model = build_model(model_name, config.scale).to(device)
        logger.info(f"Model parameters: {count_parameters(model):,}")

        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.lr,
            betas=(0.9, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.training.lr_step,
            gamma=config.training.lr_gamma,
        )

        csv_logger = CSVLogger(
            experiment_dir / "train_log.csv",
            ["epoch", "train_loss", "val_psnr", "val_ssim", "val_lpips", "lr"],
        )

        # Load best PSNR from previous runs
        best_metrics_path = experiment_dir / "best_metrics.json"
        best_psnr = -1.0
        if best_metrics_path.exists():
            import json
            try:
                data = json.loads(best_metrics_path.read_text(encoding="utf-8"))
                best_psnr = float(data.get("val_psnr", -1.0))
                logger.info(f"Loaded previous best PSNR: {best_psnr:.2f}")
            except Exception as e:
                logger.warning(f"Could not load best metrics: {e}")

        # Load best SSIM from previous runs (companion checkpoint).
        best_ssim = -1.0
        best_ssim_metrics_path = experiment_dir / "best_ssim_metrics.json"
        if best_ssim_metrics_path.exists():
            import json
            try:
                data = json.loads(best_ssim_metrics_path.read_text(encoding="utf-8"))
                best_ssim = float(data.get("val_ssim", -1.0))
                logger.info(f"Loaded previous best SSIM: {best_ssim:.4f}")
            except Exception as e:
                logger.warning(f"Could not load best SSIM metrics: {e}")

        # Resume from checkpoint
        start_epoch = 1
        last_checkpoint_path = checkpoints_dir / "last.pth"
        if config.training.resume and last_checkpoint_path.exists():
            try:
                start_epoch = CheckpointManager.load_training_state(
                    model, optimizer, last_checkpoint_path, device
                )
            except CheckpointError as e:
                logger.warning(f"Could not resume from checkpoint: {e}")

        if start_epoch > config.training.epochs:
            logger.info("Training already completed for configured epochs.")
            return

        # Early stopping setup
        early_stopping_patience = config.training.early_stopping_patience
        epochs_without_improvement = 0
        eval_every_epochs = int(config.training.eval_every_epochs or 1)
        sample_every_epochs = int(config.training.sample_every_epochs or 1)

        logger.info(f"Starting training from epoch {start_epoch}/{config.training.epochs}")
        logger.info(
            f"Validation cadence: every {eval_every_epochs} epoch(s); "
            f"sample cadence: every {sample_every_epochs} epoch(s)"
        )

        for epoch in range(start_epoch, config.training.epochs + 1):
            model.train()
            running_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.training.epochs}")
            first_batch = None

            try:
                for lr, hr in progress_bar:
                    lr = lr.to(device, non_blocking=True)
                    hr = hr.to(device, non_blocking=True)

                    optimizer.zero_grad()
                    sr = model(lr)
                    loss = criterion(sr, hr)
                    loss.backward()
                    
                    # Gradient clipping
                    if config.training.gradient_clip_value:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            config.training.gradient_clip_value
                        )
                    
                    optimizer.step()

                    running_loss += loss.item()
                    if first_batch is None:
                        first_batch = (lr.detach().cpu(), sr.detach().cpu(), hr.detach().cpu())
                    
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                    
                    # Check for NaN
                    if not torch.isfinite(torch.tensor(loss.item())):
                        raise RuntimeError("NaN loss detected during training!")

            except Exception as e:
                logger.error(f"Error during epoch {epoch}: {e}")
                raise

            scheduler.step()
            train_loss = running_loss / len(train_loader)
            run_validation = _should_run_periodic(epoch, eval_every_epochs, config.training.epochs)
            val_metrics = evaluate(model, val_loader, device, model_name, compute_lpips=compute_lpips) if run_validation else None

            csv_logger.write(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_psnr": val_metrics["psnr"] if val_metrics is not None else "",
                    "val_ssim": val_metrics["ssim"] if val_metrics is not None else "",
                    "val_lpips": val_metrics.get("lpips", "") if val_metrics is not None else "",
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            if first_batch is not None and _should_run_periodic(epoch, sample_every_epochs, config.training.epochs):
                save_comparison(*first_batch, path=samples_dir / f"epoch_{epoch:03d}.png")

            # Save last checkpoint atomically
            try:
                CheckpointManager.save_atomic(
                    checkpoints_dir / "last.pth",
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                    }
                )
            except CheckpointError as e:
                logger.error(f"Failed to save checkpoint: {e}")

            # Save best model
            if val_metrics is not None:
                if val_metrics["psnr"] > best_psnr:
                    best_psnr = val_metrics["psnr"]
                    epochs_without_improvement = 0
                    try:
                        CheckpointManager.save_atomic(
                            checkpoints_dir / "best.pth",
                            {
                                "epoch": epoch,
                                "model_state": model.state_dict(),
                                "optimizer_state": optimizer.state_dict(),
                            }
                        )
                        save_json(
                            experiment_dir / "best_metrics.json",
                            {
                                "epoch": epoch,
                                "val_psnr": val_metrics["psnr"],
                                "val_ssim": val_metrics["ssim"],
                                "model_name": model_name,
                            },
                        )
                        logger.info(f"New best PSNR: {val_metrics['psnr']:.2f}")
                    except Exception as e:
                        logger.error(f"Failed to save best model: {e}")
                else:
                    epochs_without_improvement += 1

                # Companion best-SSIM checkpoint (perceptual selection criterion).
                if val_metrics["ssim"] > best_ssim:
                    best_ssim = val_metrics["ssim"]
                    try:
                        CheckpointManager.save_atomic(
                            checkpoints_dir / "best_ssim.pth",
                            {
                                "epoch": epoch,
                                "model_state": model.state_dict(),
                                "optimizer_state": optimizer.state_dict(),
                            }
                        )
                        save_json(
                            experiment_dir / "best_ssim_metrics.json",
                            {
                                "epoch": epoch,
                                "val_psnr": val_metrics["psnr"],
                                "val_ssim": val_metrics["ssim"],
                                "model_name": model_name,
                                "selection": "best_ssim",
                            },
                        )
                        logger.info(f"New best SSIM: {val_metrics['ssim']:.4f}")
                    except Exception as e:
                        logger.error(f"Failed to save best-SSIM checkpoint: {e}")

                logger.info(
                    f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                    f"val_psnr={val_metrics['psnr']:.2f}, val_ssim={val_metrics['ssim']:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                    f"validation skipped (eval_every_epochs={eval_every_epochs})"
                )

            # Early stopping
            if (
                val_metrics is not None
                and early_stopping_patience > 0
                and epochs_without_improvement >= early_stopping_patience
            ):
                logger.info(f"Early stopping after {epochs_without_improvement} epochs without improvement")
                break

        logger.info("Training completed successfully")
        DeviceManager.cleanup()

    except SuperResolutionException as e:
        logger.error(f"Training failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}")
        raise


def d_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Discriminator hinge loss."""
    real_loss = torch.mean(torch.relu(1.0 - real_logits))
    fake_loss = torch.mean(torch.relu(1.0 + fake_logits))
    return real_loss + fake_loss


def g_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Generator hinge loss."""
    return -torch.mean(fake_logits)


def train_esrgan(config, compute_lpips: bool = False):
    """Train ESRGAN model with generator and discriminator."""
    global logger
    
    try:
        logger.info(f"Starting ESRGAN training for {config.experiment_name}")
        set_seed(config.seed)
        device = DeviceManager.setup(config.model_dump())

        experiment_dir = ensure_dir(Path(config.save_dir) / config.experiment_name)
        checkpoints_dir = ensure_dir(experiment_dir / "checkpoints")
        samples_dir = ensure_dir(experiment_dir / "samples")
        logs_dir = ensure_dir(experiment_dir / "logs")
        
        # Setup logging
        run_logger = setup_logging(logs_dir, name=f"sr_project_{config.experiment_name}")
        
        shutil.copy2(config.config_path, experiment_dir / "used_config.yaml")
        logger.info(f"Experiment dir: {experiment_dir}")

        # Create datasets
        train_dataset = SuperResolutionDataset(
            hr_dir=config.dataset.train_hr_dir,
            scale=config.scale,
            patch_size=config.dataset.patch_size,
            is_train=True,
            srcnn_mode=False,
            degradation_config=config.dataset.degradation,
            image_mode=config.dataset.image_mode,
        )
        val_dataset = SuperResolutionDataset(
            hr_dir=config.dataset.val_hr_dir,
            scale=config.scale,
            patch_size=config.dataset.patch_size,
            is_train=False,
            srcnn_mode=False,
            degradation_config={**config.dataset.degradation, "eval_crop_size": config.dataset.eval_crop_size},
            image_mode=config.dataset.image_mode,
        )
        original_val_len = len(val_dataset)
        val_dataset = _limit_validation_dataset(val_dataset, int(config.training.max_val_images or 0))
        pin_memory = _pin_memory_enabled(config, device)
        persistent_workers = _persistent_workers_enabled(config)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=max(0, config.training.num_workers // 2),
            pin_memory=pin_memory,
            persistent_workers=bool(persistent_workers and max(0, config.training.num_workers // 2) > 0),
        )
        if len(val_dataset) != original_val_len:
            logger.info(f"Validation subset enabled: {len(val_dataset)}/{original_val_len} images")

        # Build models
        generator = build_model("esrgan", config.scale).to(device)
        discriminator = Discriminator().to(device)

        logger.info(f"Generator parameters: {count_parameters(generator):,}")
        logger.info(f"Discriminator parameters: {count_parameters(discriminator):,}")

        # Load pretrained generator if specified
        if config.training.pretrained_generator:
            try:
                CheckpointManager.load_model_state(
                    generator,
                    Path(config.training.pretrained_generator),
                    device,
                    strict=False
                )
                logger.info(f"Loaded pretrained generator")
            except CheckpointError as e:
                logger.warning(f"Could not load pretrained generator: {e}")

        # Loss functions
        pixel_loss = CharbonnierLoss()
        perceptual_loss_net = VGGFeatureExtractor().to(device)
        
        # Stable-GAN: discriminator LR = generator LR / 5
        stable_gan = bool(config.training.stable_gan)
        lr_g = config.training.lr_g
        lr_d = config.training.lr_d / 5.0 if stable_gan else config.training.lr_d
        disc_warmup_epochs = int(config.training.disc_warmup_epochs) if stable_gan else 0
        r1_weight = float(config.training.r1_penalty_weight) if stable_gan else 0.0
        if stable_gan:
            logger.info(
                f"Stable-GAN: lr_g={lr_g:.2e}, lr_d={lr_d:.2e}, "
                f"disc_warmup={disc_warmup_epochs}, r1_weight={r1_weight}"
            )

        # Optimizers with learning rate schedulers
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.9, 0.99))
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.9, 0.99))
        
        scheduler_g = torch.optim.lr_scheduler.StepLR(
            optimizer_g,
            step_size=config.training.lr_step,
            gamma=config.training.lr_gamma,
        )
        scheduler_d = torch.optim.lr_scheduler.StepLR(
            optimizer_d,
            step_size=config.training.lr_step,
            gamma=config.training.lr_gamma,
        )

        csv_logger = CSVLogger(
            experiment_dir / "train_log.csv",
            ["epoch", "g_total", "g_pixel", "g_perceptual", "g_adv", "d_loss",
             "g_grad_norm", "d_grad_norm", "val_psnr", "val_ssim", "val_lpips", "lr_g", "lr_d"],
        )

        # Load best metrics
        best_psnr = -1.0
        best_metrics_path = experiment_dir / "best_metrics.json"
        if best_metrics_path.exists():
            import json
            try:
                data = json.loads(best_metrics_path.read_text(encoding="utf-8"))
                best_psnr = float(data.get("val_psnr", -1.0))
                logger.info(f"Loaded previous best PSNR: {best_psnr:.2f}")
            except Exception as e:
                logger.warning(f"Could not load best metrics: {e}")

        # Load best SSIM (gated to post-GAN epochs — perceptual checkpoint).
        best_ssim = -1.0
        best_ssim_metrics_path = experiment_dir / "best_ssim_metrics.json"
        if best_ssim_metrics_path.exists():
            import json
            try:
                data = json.loads(best_ssim_metrics_path.read_text(encoding="utf-8"))
                best_ssim = float(data.get("val_ssim", -1.0))
                logger.info(f"Loaded previous best post-GAN SSIM: {best_ssim:.4f}")
            except Exception as e:
                logger.warning(f"Could not load best SSIM metrics: {e}")

        # Resume  from checkpoint
        start_epoch = 1
        if config.training.resume:
            last_gen_path = checkpoints_dir / "last_generator.pth"
            last_dis_path = checkpoints_dir / "last_discriminator.pth"
            
            if last_gen_path.exists() and last_dis_path.exists():
                try:
                    start_epoch = CheckpointManager.load_training_state(
                        generator, optimizer_g, last_gen_path, device
                    )
                    CheckpointManager.load_training_state(
                        discriminator, optimizer_d, last_dis_path, device
                    )
                    logger.info(f"Resumed from checkpoints at epoch {start_epoch}")
                except CheckpointError as e:
                    logger.warning(f"Could not resume: {e}")

        gan_start_epoch = config.training.gan_start_epoch
        if start_epoch > config.training.epochs:
            logger.info("Training already completed.")
            return

        # Early stopping
        early_stopping_patience = config.training.early_stopping_patience
        epochs_without_improvement = 0
        eval_every_epochs = int(config.training.eval_every_epochs or 1)
        sample_every_epochs = int(config.training.sample_every_epochs or 1)

        logger.info(f"Starting ESRGAN training from epoch {start_epoch}/{config.training.epochs}")
        logger.info(f"GAN will start at epoch {gan_start_epoch}")
        logger.info(
            f"Validation cadence: every {eval_every_epochs} epoch(s); "
            f"sample cadence: every {sample_every_epochs} epoch(s)"
        )

        for epoch in range(start_epoch, config.training.epochs + 1):
            generator.train()
            discriminator.train()

            g_total_running = 0.0
            g_pixel_running = 0.0
            g_perc_running = 0.0
            g_adv_running = 0.0
            d_running = 0.0
            g_grad_norm_running = 0.0
            d_grad_norm_running = 0.0
            first_batch = None

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.training.epochs}")
            
            try:
                for lr, hr in progress_bar:
                    lr = lr.to(device, non_blocking=True)
                    hr = hr.to(device, non_blocking=True)

                    sr = generator(lr)

                    # Pixel loss
                    pixel = pixel_loss(sr, hr) * config.training.pixel_weight

                    # Perceptual loss
                    sr_features = perceptual_loss_net(sr)
                    hr_features = perceptual_loss_net(hr).detach()
                    perceptual = torch.mean(torch.abs(sr_features - hr_features)) * config.training.perceptual_weight

                    adv = torch.tensor(0.0, device=device)
                    d_loss = torch.tensor(0.0, device=device)

                    # Disc-only warmup (stable_gan): train D alone for disc_warmup_epochs
                    # before the adversarial generator loss is activated.
                    in_disc_warmup = stable_gan and epoch < gan_start_epoch + disc_warmup_epochs

                    if epoch >= gan_start_epoch:
                        # Discriminator step
                        optimizer_d.zero_grad()
                        real_logits = discriminator(hr)
                        fake_logits = discriminator(sr.detach())
                        d_loss = d_hinge_loss(real_logits, fake_logits)

                        if r1_weight > 0.0:
                            # R1 gradient penalty: penalises D gradient w.r.t. real samples
                            hr_r1 = hr.detach().requires_grad_(True)
                            real_logits_r1 = discriminator(hr_r1)
                            r1_grads = torch.autograd.grad(
                                outputs=real_logits_r1.sum(),
                                inputs=hr_r1,
                                create_graph=True,
                            )[0]
                            r1_penalty = r1_grads.pow(2).reshape(hr_r1.shape[0], -1).sum(1).mean()
                            d_loss = d_loss + (r1_weight / 2.0) * r1_penalty

                        d_loss.backward()

                        if config.training.gradient_clip_value:
                            torch.nn.utils.clip_grad_norm_(
                                discriminator.parameters(),
                                config.training.gradient_clip_value
                            )

                        d_gnorm = sum(
                            p.grad.norm().item() ** 2
                            for p in discriminator.parameters()
                            if p.grad is not None
                        ) ** 0.5
                        optimizer_d.step()

                        if in_disc_warmup:
                            # During D warmup, update G with reconstruction losses only
                            optimizer_g.zero_grad()
                            total_g = pixel + perceptual
                            total_g.backward()
                        else:
                            # Full GAN: update G with adversarial loss
                            optimizer_g.zero_grad()
                            fake_logits_g = discriminator(sr)
                            adv = g_hinge_loss(fake_logits_g) * config.training.adv_weight
                            total_g = pixel + perceptual + adv
                            total_g.backward()

                        if config.training.gradient_clip_value:
                            torch.nn.utils.clip_grad_norm_(
                                generator.parameters(),
                                config.training.gradient_clip_value
                            )

                        g_gnorm = sum(
                            p.grad.norm().item() ** 2
                            for p in generator.parameters()
                            if p.grad is not None
                        ) ** 0.5
                        optimizer_g.step()
                    else:
                        # Warmup phase (no adversarial loss)
                        optimizer_g.zero_grad()
                        total_g = pixel + perceptual
                        total_g.backward()

                        if config.training.gradient_clip_value:
                            torch.nn.utils.clip_grad_norm_(
                                generator.parameters(),
                                config.training.gradient_clip_value
                            )

                        g_gnorm = sum(
                            p.grad.norm().item() ** 2
                            for p in generator.parameters()
                            if p.grad is not None
                        ) ** 0.5
                        d_gnorm = 0.0
                        optimizer_g.step()

                    g_total_running += float(total_g.item())
                    g_pixel_running += float(pixel.item())
                    g_perc_running += float(perceptual.item())
                    g_adv_running += float(adv.item())
                    d_running += float(d_loss.item())
                    g_grad_norm_running += float(g_gnorm)
                    d_grad_norm_running += float(d_gnorm)

                    if first_batch is None:
                        first_batch = (lr.detach().cpu(), sr.detach().cpu(), hr.detach().cpu())

                    if epoch < gan_start_epoch:
                        phase = "warmup"
                    elif in_disc_warmup:
                        phase = "d-warm"
                    else:
                        phase = "gan"
                    progress_bar.set_postfix(
                        g=f"{total_g.item():.4f}",
                        d=f"{d_loss.item():.4f}",
                        phase=phase,
                    )

                    # Check for NaN
                    if not torch.isfinite(torch.tensor([total_g.item(), d_loss.item()])).all():
                        raise RuntimeError("NaN detected in losses!")

            except Exception as e:
                logger.error(f"Error during epoch {epoch}: {e}")
                raise

            scheduler_g.step()
            scheduler_d.step()
            run_validation = _should_run_periodic(epoch, eval_every_epochs, config.training.epochs)
            val_metrics = evaluate(generator, val_loader, device, "esrgan", compute_lpips=compute_lpips) if run_validation else None

            n_batches = len(train_loader)
            csv_logger.write(
                {
                    "epoch": epoch,
                    "g_total": g_total_running / n_batches,
                    "g_pixel": g_pixel_running / n_batches,
                    "g_perceptual": g_perc_running / n_batches,
                    "g_adv": g_adv_running / n_batches,
                    "d_loss": d_running / n_batches,
                    "g_grad_norm": g_grad_norm_running / n_batches,
                    "d_grad_norm": d_grad_norm_running / n_batches,
                    "val_psnr": val_metrics["psnr"] if val_metrics is not None else "",
                    "val_ssim": val_metrics["ssim"] if val_metrics is not None else "",
                    "val_lpips": val_metrics.get("lpips", "") if val_metrics is not None else "",
                    "lr_g": optimizer_g.param_groups[0]["lr"],
                    "lr_d": optimizer_d.param_groups[0]["lr"],
                }
            )

            if first_batch is not None and _should_run_periodic(epoch, sample_every_epochs, config.training.epochs):
                save_comparison(*first_batch, path=samples_dir / f"epoch_{epoch:03d}.png")

            # Save checkpoints atomically
            try:
                CheckpointManager.save_atomic(
                    checkpoints_dir / "last_generator.pth",
                    {
                        "epoch": epoch,
                        "model_state": generator.state_dict(),
                        "optimizer_state": optimizer_g.state_dict(),
                    }
                )
                CheckpointManager.save_atomic(
                    checkpoints_dir / "last_discriminator.pth",
                    {
                        "epoch": epoch,
                        "model_state": discriminator.state_dict(),
                        "optimizer_state": optimizer_d.state_dict(),
                    }
                )
            except CheckpointError as e:
                logger.error(f"Failed to save checkpoints: {e}")

            # Save best model
            if val_metrics is not None:
                if val_metrics["psnr"] > best_psnr:
                    best_psnr = val_metrics["psnr"]
                    epochs_without_improvement = 0
                    try:
                        CheckpointManager.save_atomic(
                            checkpoints_dir / "best_generator.pth",
                            {
                                "epoch": epoch,
                                "model_state": generator.state_dict(),
                                "optimizer_state": optimizer_g.state_dict(),
                            }
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
                        logger.info(f"New best PSNR: {val_metrics['psnr']:.2f}")
                    except Exception as e:
                        logger.error(f"Failed to save best model: {e}")
                else:
                    epochs_without_improvement += 1

                # Also track best post-GAN SSIM — the honest "perceptual" pick
                # for an ESRGAN-style model, since PSNR drops once the GAN term
                # engages but perceptual quality (SSIM/sharpness) keeps improving.
                if epoch >= gan_start_epoch and val_metrics["ssim"] > best_ssim:
                    best_ssim = val_metrics["ssim"]
                    try:
                        CheckpointManager.save_atomic(
                            checkpoints_dir / "best_ssim_generator.pth",
                            {
                                "epoch": epoch,
                                "model_state": generator.state_dict(),
                                "optimizer_state": optimizer_g.state_dict(),
                            }
                        )
                        save_json(
                            experiment_dir / "best_ssim_metrics.json",
                            {
                                "epoch": epoch,
                                "val_psnr": val_metrics["psnr"],
                                "val_ssim": val_metrics["ssim"],
                                "model_name": "esrgan",
                                "selection": "best_post_gan_ssim",
                                "gan_start_epoch": gan_start_epoch,
                            },
                        )
                        logger.info(f"New best post-GAN SSIM: {val_metrics['ssim']:.4f}")
                    except Exception as e:
                        logger.error(f"Failed to save best-SSIM checkpoint: {e}")

                logger.info(
                    f"Epoch {epoch}: "
                    f"g_total={g_total_running / len(train_loader):.4f}, "
                    f"d_loss={d_running / len(train_loader):.4f}, "
                    f"val_psnr={val_metrics['psnr']:.2f}, val_ssim={val_metrics['ssim']:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch}: "
                    f"g_total={g_total_running / len(train_loader):.4f}, "
                    f"d_loss={d_running / len(train_loader):.4f}, "
                    f"validation skipped (eval_every_epochs={eval_every_epochs})"
                )

            # Early stopping
            if (
                val_metrics is not None
                and early_stopping_patience > 0
                and epochs_without_improvement >= early_stopping_patience
            ):
                logger.info(f"Early stopping after {epochs_without_improvement} epochs without improvement")
                break

        logger.info("Training completed successfully")
        DeviceManager.cleanup()

    except SuperResolutionException as e:
        logger.error(f"Training failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}")
        raise


def main():
    """Main training script."""
    global logger

    parser = argparse.ArgumentParser(description="Train super-resolution models")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--stable-gan",
        action="store_true",
        default=None,
        help="Enable stable GAN schedule: lr_d=lr_g/5, disc warmup, R1 penalty",
    )
    parser.add_argument(
        "--compute-lpips",
        action="store_true",
        default=False,
        help="Compute LPIPS during validation (slower; requires lpips package)",
    )
    args = parser.parse_args()

    try:
        # Load and validate config
        config = load_config(args.config)

        # CLI --stable-gan overrides config field
        if args.stable_gan is not None:
            object.__setattr__(config.training, "stable_gan", args.stable_gan)

        #  Setup logging
        log_dir = Path(config.save_dir) / config.experiment_name / "logs"
        logger = setup_logging(log_dir, name="sr_project")
        logger.info(f"Loaded config: {args.config}")
        logger.info(f"Model: {config.model_name}, Scale: {config.scale}")
        if config.training.stable_gan:
            logger.info("Stable-GAN schedule ENABLED")
        if args.compute_lpips:
            logger.info("LPIPS validation metric ENABLED")

        # Train appropriate model
        if config.model_name.lower() == "esrgan":
            train_esrgan(config, compute_lpips=args.compute_lpips)
        else:
            train_reconstruction(config, compute_lpips=args.compute_lpips)

        logger.info("✓ All training completed successfully!")

    except Exception as e:
        if logger:
            logger.critical(f"Fatal error: {e}")
        else:
            print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
