"""
train_hsi_3dcnn.py
──────────────────
Train HSI 3D CNN U-Net.

CLI:
    neon-train-hsi-cnn --preprocessed-dir data/hsi_preprocessed
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path

import click
import torch
import wandb
from loguru import logger

from neon_tree_crown.data.datasets import build_hsi_dataloaders
from neon_tree_crown.models.hsi_3dcnn import HSI3DUNet
from neon_tree_crown.models.losses import CombinedLoss
from neon_tree_crown.training.engine import EarlyStopping, train_one_epoch, validate
from neon_tree_crown.training.wandb_utils import (
    build_warmup_cosine_scheduler,
    log_epoch_metrics,
    log_val_predictions,
)

SEED = 42


def _set_seed(seed=SEED):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@click.command()
@click.option("--preprocessed-dir", required=True, type=click.Path())
@click.option("--save-dir",         default=None)
@click.option("--base-channels",    default=8,    show_default=True, type=int)
@click.option("--batch-size",       default=4,    show_default=True, type=int)
@click.option("--num-workers",      default=4,    show_default=True, type=int)
@click.option("--epochs",           default=100,  show_default=True, type=int)
@click.option("--lr",               default=3e-4, show_default=True, type=float)
@click.option("--weight-decay",     default=0.01, show_default=True, type=float)
@click.option("--patience",         default=15,   show_default=True, type=int)
@click.option("--wandb-project",    default="neon-tree-crown", show_default=True)
@click.option("--wandb-offline",    is_flag=True)
def main(
    preprocessed_dir, save_dir, base_channels, batch_size, num_workers,
    epochs, lr, weight_decay, patience, wandb_project, wandb_offline,
):
    """Train HSI 3D CNN U-Net."""
    _set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    run_id  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(save_dir) if save_dir else Path("runs") / f"{run_id}_hsi_3dcnn"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(preprocessed_dir=preprocessed_dir, batch_size=batch_size, num_workers=num_workers)

    if wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    wandb.init(project=wandb_project, name=f"{run_id}_hsi_3dcnn", config=cfg)

    train_loader, val_loader, n_bands = build_hsi_dataloaders(cfg)
    logger.info(f"n_bands (preprocessed) = {n_bands}")

    model     = HSI3DUNet(n_bands=n_bands, base_channels=base_channels).to(device)
    criterion = CombinedLoss(gamma=2.0, pos_weight=2.0, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = build_warmup_cosine_scheduler(optimizer, total_epochs=epochs)

    early    = EarlyStopping(patience=patience)
    best_iou = 0.0

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl = validate(model, val_loader, criterion, device)
        sched.step()
        log_epoch_metrics(epoch, tr, vl, optimizer)
        log_val_predictions(epoch, model, val_loader, device, modality="hsi")
        logger.info(
            f"E{epoch:03d} | loss {tr['train_loss']:.4f}/{vl['val_loss']:.4f} | "
            f"IoU {tr['train_iou']:.3f}/{vl['val_iou']:.3f} | "
            f"Dice {tr['train_dice']:.3f}/{vl['val_dice']:.3f} | "
            f"LR {optimizer.param_groups[0]['lr']:.2e}"
        )

        if vl["val_iou"] > best_iou:
            best_iou = vl["val_iou"]
            torch.save({"model_state": model.state_dict(), "n_bands": n_bands,
                        "base_channels": base_channels, "epoch": epoch},
                       run_dir / "best_model.pth")
            logger.info(f"  ✓ Saved (IoU={best_iou:.4f})")

        if early.step(vl["val_iou"]):
            logger.warning("Early stopping.")
            break

    logger.success(f"Done. Best val IoU = {best_iou:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
