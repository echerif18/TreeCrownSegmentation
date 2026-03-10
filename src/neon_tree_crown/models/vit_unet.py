"""
ViT U-Net for RGB and HSI modalities.
  - RGBViTUNet   : standard ViT encoder + CNN decoder
  - HSIViTUNet   : SpectralProjection → ViT encoder + CNN decoder
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ─────────────────────────────────────────────────────────────────────────────
# Shared ViT building blocks
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 320, patch_size: int = 16, in_ch: int = 3, hidden: int = 384):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches  = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, hidden, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        B, D, H, W = x.shape
        return x.flatten(2).transpose(1, 2), H, W   # (B, N, D), H, W


class MHSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.heads   = heads
        self.scale   = (dim // heads) ** -0.5
        self.qkv     = nn.Linear(dim, dim * 3, bias=False)
        self.proj    = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H       = self.heads
        head_d  = D // H
        qkv = self.qkv(x).reshape(B, N, 3, H, head_d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        return (attn @ v).transpose(1, 2).reshape(B, N, D)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MHSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ConvUp(nn.Module):
    """Upsample + double-conv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


# ─────────────────────────────────────────────────────────────────────────────
# RGB ViT U-Net
# ─────────────────────────────────────────────────────────────────────────────

class RGBViTUNet(nn.Module):
    def __init__(
        self,
        img_size:    int   = 320,
        patch_size:  int   = 16,
        in_ch:       int   = 3,
        hidden:      int   = 384,
        depth:       int   = 6,
        heads:       int   = 8,
        mlp_dim:     int   = 1536,
        drop:        float = 0.1,
        out_channels: int  = 1,
    ):
        super().__init__()
        self.patch_size = patch_size
        n = img_size // patch_size

        self.patch_embed  = PatchEmbed(img_size, patch_size, in_ch, hidden)
        self.cls_token    = nn.Parameter(torch.zeros(1, 1, hidden))
        self.pos_embed    = nn.Parameter(torch.zeros(1, n * n + 1, hidden))
        self.pos_drop     = nn.Dropout(drop)
        self.transformer  = nn.Sequential(*[TransformerBlock(hidden, heads, mlp_dim, drop) for _ in range(depth)])
        self.norm         = nn.LayerNorm(hidden)

        # Decoder
        self.dec4 = ConvUp(hidden,       hidden // 2)
        self.dec3 = ConvUp(hidden // 2,  hidden // 4)
        self.dec2 = ConvUp(hidden // 4,  hidden // 8)
        self.dec1 = ConvUp(hidden // 8,  hidden // 16)
        self.head = nn.Conv2d(hidden // 16, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tokens, H, W = self.patch_embed(x)                 # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed[:, :tokens.shape[1]])
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        patch_tokens = tokens[:, 1:].transpose(1, 2).reshape(B, -1, H, W)  # (B,D,H,W)
        d4 = self.dec4(patch_tokens)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        return F.interpolate(self.head(d1), size=x.shape[2:], mode="bilinear", align_corners=False)


# ─────────────────────────────────────────────────────────────────────────────
# HSI ViT U-Net
# ─────────────────────────────────────────────────────────────────────────────

class SpectralProjection(nn.Module):
    """Collapse any number of bands → fixed (spectral_out, H, W)."""
    def __init__(self, spectral_out: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LazyConv2d(128, 1, bias=False), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, spectral_out, 1, bias=False), nn.BatchNorm2d(spectral_out), nn.GELU(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class HSIViTUNet(nn.Module):
    def __init__(
        self,
        n_bands:     int   = 356,
        spectral_out: int  = 64,
        img_size:    int   = 32,
        patch_size:  int   = 4,
        hidden:      int   = 256,
        depth:       int   = 4,
        heads:       int   = 8,
        mlp_dim:     int   = 1024,
        drop:        float = 0.1,
        out_channels: int  = 1,
    ):
        super().__init__()
        self.spectral = SpectralProjection(spectral_out)
        self.vit_rgb  = RGBViTUNet(img_size, patch_size, spectral_out, hidden,
                                    depth, heads, mlp_dim, drop, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit_rgb(self.spectral(x))