"""HSI 3D CNN U-Net for hyperspectral tree-crown segmentation."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DBlock(nn.Module):
    """3D Conv → BN → ReLU ×2 with spectral kernel."""
    def __init__(self, in_ch: int, out_ch: int, spectral_k: int = 7, dropout: float = 0.1):
        super().__init__()
        pad = (spectral_k // 2, 1, 1)
        self.block = nn.Sequential(
            nn.Conv3d(in_ch,  out_ch, (spectral_k, 3, 3), padding=pad, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True), nn.Dropout3d(dropout),
            nn.Conv3d(out_ch, out_ch, (spectral_k, 3, 3), padding=pad, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class HSI3DUNet(nn.Module):
    """
    Input : (B, n_bands, H, W)  — reshaped internally to (B, 1, n_bands, H, W)
    Output: (B, 1, H, W)        — binary logit map
    """
    def __init__(self, n_bands: int = 356, base_channels: int = 8):
        super().__init__()
        b = base_channels

        # Encoder
        self.enc1 = Conv3DBlock(1, b)
        self.enc2 = Conv3DBlock(b, b * 2)
        self.pool = nn.MaxPool3d((2, 2, 2))

        # Bottleneck
        self.bottleneck = Conv3DBlock(b * 2, b * 4)

        # Collapse spectral dim → 2D feature map
        self.spectral_collapse = nn.AdaptiveAvgPool3d((1, None, None))

        # 2D decoder
        self.up2  = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(b * 2, b * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(b * 2), nn.ReLU(inplace=True),
        )
        self.up1  = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(b, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, 1, C, H, W)
        x = x.unsqueeze(1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        bn = self.bottleneck(self.pool(e2))

        # Collapse spectral → 2D
        f  = self.spectral_collapse(bn).squeeze(2)   # (B, ch, H, W)

        d2 = self.dec2(self.up2(f))
        d1 = self.dec1(self.up1(d2))
        return F.interpolate(self.head(d1), size=x.shape[3:], mode="bilinear", align_corners=False)