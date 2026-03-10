from __future__ import annotations

import numpy as np
import torch
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


RGB_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
CLASS_LABELS = {0: "bg", 1: "tree"}


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int = 5,
    start_factor: float = 0.1,
    eta_min: float = 1e-6,
):
    warmup_epochs = min(warmup_epochs, max(1, total_epochs))
    warmup = LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=eta_min)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])


def log_epoch_metrics(
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    optimizer: torch.optim.Optimizer,
    phase: str | None = None,
) -> None:
    payload = {"epoch": epoch}
    if phase:
        payload["phase"] = phase

    for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
        for key, value in metrics.items():
            short_key = key[len(prefix) + 1:] if key.startswith(f"{prefix}_") else key
            payload[f"{prefix}/{short_key}"] = value

    for idx, group in enumerate(optimizer.param_groups):
        payload[f"lr/group_{idx}"] = group["lr"]
    if optimizer.param_groups:
        payload["lr"] = optimizer.param_groups[0]["lr"]

    wandb.log(payload)


def log_val_predictions(
    epoch: int,
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    modality: str,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    max_items: int = 4,
) -> None:
    if epoch % 10 != 0:
        return

    was_training = model.training
    model.eval()
    images, masks = next(iter(val_loader))

    with torch.no_grad():
        logits = model(images.to(device, non_blocking=True))
        preds = (torch.sigmoid(logits) > 0.5).float().cpu()

    if modality == "rgb":
        mean = RGB_MEAN if mean is None else mean
        std = RGB_STD if std is None else std
        base_images = _rgb_previews(images.cpu(), mean, std)
    elif modality == "hsi":
        base_images = _hsi_previews(images.cpu())
    else:
        raise ValueError(f"Unknown preview modality: {modality}")

    samples = []
    limit = min(max_items, len(images))
    for idx in range(limit):
        samples.append(
            wandb.Image(
                base_images[idx],
                masks={
                    "ground_truth": {"mask_data": masks[idx, 0].cpu().numpy(), "class_labels": CLASS_LABELS},
                    "prediction": {"mask_data": preds[idx, 0].numpy(), "class_labels": CLASS_LABELS},
                },
                caption=f"Epoch {epoch}",
            )
        )

    wandb.log({"val/predictions": samples, "epoch": epoch})

    if was_training:
        model.train()


def _rgb_previews(images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> list[np.ndarray]:
    mean = mean.cpu()
    std = std.cpu()
    previews = []
    for image in images:
        rgb = (image * std + mean).permute(1, 2, 0).clamp(0, 1).numpy()
        previews.append(rgb)
    return previews


def _hsi_previews(cubes: torch.Tensor) -> list[np.ndarray]:
    previews = []
    n_bands = cubes.shape[1]
    wavelengths = np.linspace(400, 2450, n_bands)
    nir_idx = int(np.argmin(np.abs(wavelengths - 850)))
    red_idx = int(np.argmin(np.abs(wavelengths - 670)))
    green_idx = int(np.argmin(np.abs(wavelengths - 550)))

    for cube in cubes.numpy():
        false_color = np.stack([cube[nir_idx], cube[red_idx], cube[green_idx]], axis=-1)
        lo, hi = np.percentile(false_color, (2, 98))
        false_color = np.clip((false_color - lo) / (hi - lo + 1e-6), 0, 1)
        previews.append(false_color)
    return previews
