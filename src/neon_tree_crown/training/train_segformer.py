"""
train_segformer.py
──────────────────
Fine-tune SegFormer-B2 for binary tree-crown segmentation with
3-phase progressive unfreezing and anti-overfitting measures.

CLI:
    neon-train-segformer --data-root data/splits/rgb_binary --epochs-p1 15 ...
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path

import click
import torch
import wandb
from loguru import logger
from transformers import AutoImageProcessor

from neon_tree_crown.data.datasets import build_segformer_dataloaders
from neon_tree_crown.models.losses import CombinedLoss
from neon_tree_crown.models.segformer import SegFormerBinary
from neon_tree_crown.training.engine import EarlyStopping, train_one_epoch, validate
from neon_tree_crown.training.wandb_utils import (
    build_warmup_cosine_scheduler,
    log_epoch_metrics,
    log_val_predictions,
)


SEED = 42


def _set_seed(seed: int = SEED):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _make_optimizer_and_scheduler(param_groups, n_epochs, warmup=5):
    opt    = torch.optim.AdamW(param_groups, weight_decay=0.05)
    return opt, build_warmup_cosine_scheduler(opt, total_epochs=n_epochs, warmup_epochs=warmup)


def run_phase(
    tag:           str,
    n_epochs:      int,
    model:         SegFormerBinary,
    optimizer:     torch.optim.Optimizer,
    scheduler,
    criterion:     CombinedLoss,
    train_loader,
    val_loader,
    device:        torch.device,
    save_dir:      Path,
    history:       dict,
    early:         EarlyStopping,
    best_iou:      float,
    global_epoch:  int,
    preview_mean:  torch.Tensor,
    preview_std:   torch.Tensor,
) -> tuple[float, bool, int]:
    """Run one training phase; return (best_iou, stopped)."""
    for epoch in range(1, n_epochs + 1):
        global_epoch += 1
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl = validate(model, val_loader, criterion, device)
        scheduler.step()

        for k, v in {**tr, **vl}.items():
            history.setdefault(k, []).append(v)

        log_epoch_metrics(global_epoch, tr, vl, optimizer, phase=tag)
        log_val_predictions(
            global_epoch,
            model,
            val_loader,
            device,
            modality="rgb",
            mean=preview_mean,
            std=preview_std,
        )

        lr_groups = [f"{pg['lr']:.2e}" for pg in optimizer.param_groups]
        logger.info(
            f"[{tag}] E{epoch:03d} | "
            f"loss {tr['train_loss']:.4f}/{vl['val_loss']:.4f} | "
            f"IoU {tr['train_iou']:.3f}/{vl['val_iou']:.3f} | "
            f"Dice {tr['train_dice']:.3f}/{vl['val_dice']:.3f} | "
            f"LR {lr_groups}"
        )

        if vl["val_iou"] > best_iou:
            best_iou = vl["val_iou"]
            model.save(save_dir / "best_model")
            logger.info(f"  ✓ New best IoU {best_iou:.4f} — saved.")

        if early.step(vl["val_iou"]):
            logger.warning(f"  Early stopping triggered at epoch {epoch}.")
            return best_iou, True, global_epoch

    return best_iou, False, global_epoch


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--data-root",    required=True, type=click.Path(), help="Path to splits/rgb_binary")
@click.option("--pretrained",   default="nvidia/segformer-b2-finetuned-ade-512-512", show_default=True)
@click.option("--save-dir",     default=None,  help="Override checkpoint dir")
@click.option("--img-size",     default=512,   show_default=True, type=int)
@click.option("--batch-size",   default=4,     show_default=True, type=int)
@click.option("--num-workers",  default=4,     show_default=True, type=int)
@click.option("--epochs-p1",    default=15,    show_default=True, type=int)
@click.option("--epochs-p2",    default=20,    show_default=True, type=int)
@click.option("--epochs-p3",    default=25,    show_default=True, type=int)
@click.option("--lr-head",      default=3e-4,  show_default=True, type=float)
@click.option("--lr-b3",        default=3e-5,  show_default=True, type=float)
@click.option("--lr-b2",        default=1e-5,  show_default=True, type=float)
@click.option("--head-dropout", default=0.3,   show_default=True, type=float)
@click.option("--patience",     default=12,    show_default=True, type=int)
@click.option("--wandb-project",default="neon-tree-crown", show_default=True)
@click.option("--wandb-offline", is_flag=True)
def main(
    data_root, pretrained, save_dir, img_size, batch_size, num_workers,
    epochs_p1, epochs_p2, epochs_p3, lr_head, lr_b3, lr_b2,
    head_dropout, patience, wandb_project, wandb_offline,
):
    """Fine-tune SegFormer-B2 with 3-phase progressive unfreezing."""
    _set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    run_id  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(save_dir) if save_dir else Path("runs") / f"{run_id}_segformer_b2"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(
        data_root=data_root, img_size=img_size,
        batch_size=batch_size, num_workers=num_workers,
    )

    if wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    wandb.init(project=wandb_project, name=f"{run_id}_segformer_b2", config=cfg)

    # ── Data ──────────────────────────────────────────────────────────────────
    processor     = AutoImageProcessor.from_pretrained(pretrained)
    train_loader, val_loader = build_segformer_dataloaders(cfg, processor)
    processor.save_pretrained(run_dir / "best_model")  # save processor alongside model
    preview_mean = torch.tensor(processor.image_mean).view(3, 1, 1)
    preview_std = torch.tensor(processor.image_std).view(3, 1, 1)

    # ── Model + loss ──────────────────────────────────────────────────────────
    model     = SegFormerBinary(pretrained, img_size, head_dropout).to(device)
    criterion = CombinedLoss(gamma=2.0, pos_weight=2.0, dice_weight=0.5)
    history   = {}
    best_iou  = 0.0
    early     = EarlyStopping(patience=patience)
    global_epoch = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: head only
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=== Phase 1: decode head only ===")
    model.freeze_all_encoder()
    opt1, sch1 = _make_optimizer_and_scheduler(
        [{"params": model.segformer.decode_head.parameters(), "lr": lr_head}],
        epochs_p1,
    )
    best_iou, stopped, global_epoch = run_phase(
        "P1", epochs_p1, model, opt1, sch1, criterion,
        train_loader, val_loader, device, run_dir, history, early, best_iou, global_epoch,
        preview_mean, preview_std,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: + block[3]
    # ─────────────────────────────────────────────────────────────────────────
    if not stopped:
        logger.info("=== Phase 2: + encoder block[3] ===")
        model.unfreeze_block3()
        early.reset_counter()
        opt2, sch2 = _make_optimizer_and_scheduler(
            [
                {"params": model.segformer.decode_head.parameters(), "lr": lr_head},
                {"params": model.segformer.segformer.encoder.block[-1].parameters(), "lr": lr_b3},
            ],
            epochs_p2,
        )
        best_iou, stopped, global_epoch = run_phase(
            "P2", epochs_p2, model, opt2, sch2, criterion,
            train_loader, val_loader, device, run_dir, history, early, best_iou, global_epoch,
            preview_mean, preview_std,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3: + block[2]
    # ─────────────────────────────────────────────────────────────────────────
    if not stopped:
        logger.info("=== Phase 3: + encoder block[2] ===")
        model.unfreeze_block2()
        early.reset_counter()
        opt3, sch3 = _make_optimizer_and_scheduler(
            [
                {"params": model.segformer.decode_head.parameters(), "lr": lr_head},
                {"params": model.segformer.segformer.encoder.block[-1].parameters(), "lr": lr_b3},
                {"params": model.segformer.segformer.encoder.block[-2].parameters(), "lr": lr_b2},
            ],
            epochs_p3,
        )
        best_iou, _, global_epoch = run_phase(
            "P3", epochs_p3, model, opt3, sch3, criterion,
            train_loader, val_loader, device, run_dir, history, early, best_iou, global_epoch,
            preview_mean, preview_std,
        )

    logger.success(f"Training complete. Best val IoU = {best_iou:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
