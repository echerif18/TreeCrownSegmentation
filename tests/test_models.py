"""Smoke tests — verify model forward passes without GPU."""
import torch
import pytest


def test_attention_unet():
    from neon_tree_crown.models.attention_unet import AttentionUNet
    model = AttentionUNet(in_channels=3, out_channels=1, base_channels=8)
    x     = torch.randn(2, 3, 64, 64)
    out   = model(x)
    assert out.shape == (2, 1, 64, 64)


def test_vit_unet_rgb():
    from neon_tree_crown.models.vit_unet import RGBViTUNet
    model = RGBViTUNet(img_size=64, patch_size=8, hidden=64, depth=2, heads=4, mlp_dim=128)
    x     = torch.randn(2, 3, 64, 64)
    out   = model(x)
    assert out.shape == (2, 1, 64, 64)


def test_hsi_3dcnn():
    from neon_tree_crown.models.hsi_3dcnn import HSI3DUNet
    model = HSI3DUNet(n_bands=32, base_channels=4)
    x     = torch.randn(2, 32, 16, 16)
    out   = model(x)
    assert out.shape[0] == 2 and out.shape[1] == 1


def test_losses():
    from neon_tree_crown.models.losses import CombinedLoss
    loss = CombinedLoss()
    logits  = torch.randn(4, 1, 32, 32)
    targets = torch.randint(0, 2, (4, 1, 32, 32)).float()
    val     = loss(logits, targets)
    assert val.item() > 0


def test_metrics():
    from neon_tree_crown.utils.metrics import compute_iou, compute_dice
    logits  = torch.ones(2, 1, 16, 16) * 3.0
    targets = torch.ones(2, 1, 16, 16)
    assert compute_iou(logits, targets) == pytest.approx(1.0, abs=0.01)
    assert compute_dice(logits, targets) == pytest.approx(1.0, abs=0.01)
