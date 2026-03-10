"""Focal + Dice combined loss — shared across all models."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.pw    = torch.tensor([pos_weight])

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pw.to(logits.device)
        )
        return ((1 - torch.exp(-bce)) ** self.gamma * bce).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(-1)
        tgts  = targets.view(-1)
        inter = (probs * tgts).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (probs.sum() + tgts.sum() + self.smooth)


class CombinedLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: float = 2.0, dice_weight: float = 0.5):
        super().__init__()
        self.focal       = FocalLoss(gamma, pos_weight)
        self.dice        = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal(logits, targets) + self.dice_weight * self.dice(logits, targets)
