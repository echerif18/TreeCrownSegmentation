"""
train_rgb_vit.py
────────────────
Train RGB ViT U-Net.

CLI:
    neon-train-vit-rgb --data-root data/splits/rgb_binary
"""
from __future__ import annotations

import datetime, os
from pathlib import Path

import click, torch, wandb
from loguru import logger

from neon_tree_crown.data.datasets import build_rgb_dataloaders
from neon_tree_crown.models.vit_unet import RGBViTUNet
from neon_tree_crown.models.losses import CombinedLoss
from neon_tree_crown.training.engine import EarlyStopping, train_one_epoch, validate
from neon_tree_crown.training.wandb_utils import (
    build_warmup_cosine_scheduler,
    log_epoch_metrics,
    log_val_predictions,
)


def _set_seed(seed=42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@click.command()
@click.option("--data-root",     required=True, type=click.Path())
@click.option("--save-dir",      default=None)
@click.option("--img-size",      default=320,   show_default=True, type=int)
@click.option("--patch-size",    default=16,    show_default=True, type=int)
@click.option("--vit-depth",     default=6,     show_default=True, type=int)
@click.option("--vit-heads",     default=8,     show_default=True, type=int)
@click.option("--vit-hidden",    default=384,   show_default=True, type=int)
@click.option("--vit-mlp-dim",   default=1536,  show_default=True, type=int)
@click.option("--dropout",       default=0.1,   show_default=True, type=float)
@click.option("--batch-size",    default=4,     show_default=True, type=int)
@click.option("--num-workers",   default=4,     show_default=True, type=int)
@click.option("--epochs",        default=100,   show_default=True, type=int)
@click.option("--lr",            default=3e-4,  show_default=True, type=float)
@click.option("--weight-decay",  default=0.05,  show_default=True, type=float)
@click.option("--patience",      default=15,    show_default=True, type=int)
@click.option("--wandb-project", default="neon-tree-crown", show_default=True)
@click.option("--wandb-offline",  is_flag=True)
def main(
    data_root, save_dir, img_size, patch_size, vit_depth, vit_heads,
    vit_hidden, vit_mlp_dim, dropout, batch_size, num_workers,
    epochs, lr, weight_decay, patience, wandb_project, wandb_offline,
):
    """Train RGB ViT U-Net."""
    _set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(save_dir) if save_dir else Path("runs") / f"{run_id}_vit_rgb"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(data_root=data_root, img_size=img_size, batch_size=batch_size, num_workers=num_workers)

    if wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    wandb.init(project=wandb_project, name=f"{run_id}_vit_rgb", config=cfg)

    train_loader, val_loader = build_rgb_dataloaders(cfg)

    model = RGBViTUNet(
        img_size=img_size, patch_size=patch_size, in_ch=3,
        hidden=vit_hidden, depth=vit_depth, heads=vit_heads,
        mlp_dim=vit_mlp_dim, drop=dropout,
    ).to(device)
    criterion = CombinedLoss(gamma=2.0, pos_weight=2.0, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = build_warmup_cosine_scheduler(optimizer, total_epochs=epochs)

    early = EarlyStopping(patience=patience)
    best_iou = 0.0

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl = validate(model, val_loader, criterion, device)
        sched.step()
        log_epoch_metrics(epoch, tr, vl, optimizer)
        log_val_predictions(epoch, model, val_loader, device, modality="rgb")
        logger.info(
            f"E{epoch:03d} | loss {tr['train_loss']:.4f}/{vl['val_loss']:.4f} | "
            f"IoU {tr['train_iou']:.3f}/{vl['val_iou']:.3f} | "
            f"Dice {tr['train_dice']:.3f}/{vl['val_dice']:.3f} | "
            f"LR {optimizer.param_groups[0]['lr']:.2e}"
        )
        if vl["val_iou"] > best_iou:
            best_iou = vl["val_iou"]
            torch.save({"model_state": model.state_dict(), "cfg": cfg},
                       run_dir / "best_model.pth")
        if early.step(vl["val_iou"]): break

    logger.success(f"Best val IoU = {best_iou:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
