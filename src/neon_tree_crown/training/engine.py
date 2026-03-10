"""
Shared training engine — train_one_epoch / validate / EarlyStopping.
"""
from __future__ import annotations
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from neon_tree_crown.utils.metrics import compute_dice, compute_f1, compute_iou


# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    clip_grad: float = 1.0,
) -> Dict[str, float]:
    model.train()
    total_loss = total_iou = total_dice = total_f1 = 0.0
    n = 0

    for imgs, masks in tqdm(loader, desc="  Train", leave=False):
        imgs  = imgs.to(device,  non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, masks)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()
        total_iou  += compute_iou(logits,  masks)
        total_dice += compute_dice(logits, masks)
        total_f1   += compute_f1(logits,   masks)
        n += 1

    return {
        "train_loss": total_loss / n,
        "train_iou":  total_iou  / n,
        "train_dice": total_dice / n,
        "train_f1":   total_f1   / n,
    }


@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = total_iou = total_dice = total_f1 = 0.0
    n = 0

    for imgs, masks in tqdm(loader, desc="  Val  ", leave=False):
        imgs  = imgs.to(device,  non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(imgs)
        loss   = criterion(logits, masks)

        total_loss += loss.item()
        total_iou  += compute_iou(logits,  masks)
        total_dice += compute_dice(logits, masks)
        total_f1   += compute_f1(logits,   masks)
        n += 1

    return {
        "val_loss": total_loss / n,
        "val_iou":  total_iou  / n,
        "val_dice": total_dice / n,
        "val_f1":   total_f1   / n,
    }


# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = -float("inf")
        self.stopped    = False

    def step(self, score: float) -> bool:
        """Return True if training should stop."""
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
        return self.stopped

    def reset_counter(self):
        self.counter = 0
