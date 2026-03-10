"""Binary segmentation metrics — IoU, Dice, F1."""
from __future__ import annotations

import torch


def compute_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds    = (torch.sigmoid(logits.detach()) > threshold).float()
    targets  = targets.detach().float()
    inter    = (preds * targets).sum()
    union    = preds.sum() + targets.sum() - inter
    return (inter / (union + 1e-6)).item()


def compute_dice(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds   = (torch.sigmoid(logits.detach()) > threshold).float()
    targets = targets.detach().float()
    inter   = (preds * targets).sum()
    return (2 * inter / (preds.sum() + targets.sum() + 1e-6)).item()


def compute_f1(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    return compute_dice(logits, targets, threshold)
