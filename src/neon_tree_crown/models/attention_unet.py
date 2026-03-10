"""Attention U-Net (Oktay et al., 2018) for RGB binary segmentation."""

# ────────────────────────────────────────────────────────────────────────────
# Attention U-Net (Oktay et al., 2018)
#
# Key idea: Attention gates on skip connections learn to FOCUS on tree regions
# and SUPPRESS background. This directly helps with class imbalance without
# needing to manually tune weights.
#
# Architecture:
#   Encoder (4 levels) → Bottleneck → Decoder (4 levels)
#   Skip connections with attention gates at each decoder level
# ────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1,    1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape != x1.shape:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        return x * self.psi(F.relu(g1 + x1, inplace=True))


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 64):
        super().__init__()
        b = base_channels

        # Encoder
        self.enc1 = ConvBlock(in_channels, b)
        self.enc2 = ConvBlock(b,     b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.enc4 = ConvBlock(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(b * 8, b * 16)

        # Decoder
        self.up4  = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.ag4  = AttentionGate(b * 8, b * 8, b * 4)
        self.dec4 = ConvBlock(b * 16, b * 8)

        self.up3  = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.ag3  = AttentionGate(b * 4, b * 4, b * 2)
        self.dec3 = ConvBlock(b * 8, b * 4)

        self.up2  = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.ag2  = AttentionGate(b * 2, b * 2, b)
        self.dec2 = ConvBlock(b * 4, b * 2)

        self.up1  = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.ag1  = AttentionGate(b,     b,     b // 2)
        self.dec1 = ConvBlock(b * 2, b)

        self.head = nn.Conv2d(b, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(bn), self.ag4(self.up4(bn), e4)], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), self.ag3(self.up3(d4), e3)], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), self.ag2(self.up2(d3), e2)], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), self.ag1(self.up1(d2), e1)], 1))
        return self.head(d1)
