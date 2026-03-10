"""
datasets.py
───────────
PyTorch Dataset classes for RGB and HSI modalities.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation factories
# ─────────────────────────────────────────────────────────────────────────────

def get_rgb_train_transforms(img_size: int = 320) -> A.Compose:
    return A.Compose([
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, fill=0),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
        A.ElasticTransform(alpha=60, sigma=6, p=0.3),
        A.RandomCrop(img_size, img_size, p=1.0),
        A.OneOf([
            A.RandomBrightnessContrast(p=1.0),
            A.HueSaturationValue(p=1.0),
            A.CLAHE(p=1.0),
        ], p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(p=0.2),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(1, 32), hole_width_range=(1, 32), p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_rgb_val_transforms(img_size: int = 320) -> A.Compose:
    return A.Compose([
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, fill=0),
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_segformer_train_transforms(img_size: int = 512) -> A.Compose:
    """Geometric-only transforms — processor handles normalize + to-tensor."""
    return A.Compose([
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, fill=0),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
        A.ElasticTransform(alpha=60, sigma=6, p=0.3),
        A.RandomCrop(img_size, img_size, p=1.0),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(1, 32), hole_width_range=(1, 32), p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
    ])


def get_segformer_val_transforms(img_size: int = 512) -> A.Compose:
    return A.Compose([
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, fill=0),
        A.CenterCrop(img_size, img_size),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# RGB Dataset (Attention U-Net / ViT U-Net)
# ─────────────────────────────────────────────────────────────────────────────

class RGBDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        mask_paths:  List[Path],
        transform:   Optional[Callable] = None,
    ):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img  = np.array(Image.open(self.image_paths[idx]).convert("RGB"))  # (H,W,3)
        mask = tifffile.imread(str(self.mask_paths[idx])).astype(np.float32)
        mask = (mask > 0).astype(np.float32)                               # binary

        if self.transform:
            out  = self.transform(image=img, mask=mask)
            img  = out["image"]
            mask = out["mask"]

        if not isinstance(img, torch.Tensor):
            img  = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0) if isinstance(mask, np.ndarray) else mask.unsqueeze(0)
        return img, mask.float()


# ─────────────────────────────────────────────────────────────────────────────
# SegFormer Dataset (uses HuggingFace AutoImageProcessor)
# ─────────────────────────────────────────────────────────────────────────────

class SegFormerDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        mask_paths:  List[Path],
        processor:   AutoImageProcessor,
        transform:   Optional[Callable] = None,
    ):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.processor   = processor
        self.transform   = transform

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img  = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = tifffile.imread(str(self.mask_paths[idx])).astype(np.float32)
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            out  = self.transform(image=img, mask=mask)
            img  = out["image"]
            mask = out["mask"]

        # Processor: resize, normalize, to tensor
        inputs = self.processor(images=Image.fromarray(img), return_tensors="pt")
        pixel  = inputs["pixel_values"].squeeze(0)          # (3, H, W)
        mask_t = torch.from_numpy(mask).unsqueeze(0).float() # (1, H, W)
        return pixel, mask_t


# ─────────────────────────────────────────────────────────────────────────────
# HSI Dataset
# ─────────────────────────────────────────────────────────────────────────────

class HSIDataset(Dataset):
    """Loads preprocessed HSI cubes (tifffile) + binary labels."""
    def __init__(
        self,
        cube_paths:  List[Path],
        mask_paths:  List[Path],
        augmentation: Optional[Callable] = None,
    ):
        self.cube_paths   = cube_paths
        self.mask_paths   = mask_paths
        self.augmentation = augmentation

    def __len__(self): return len(self.cube_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cube = tifffile.imread(str(self.cube_paths[idx])).astype(np.float32)  # (B,H,W)
        mask = tifffile.imread(str(self.mask_paths[idx])).astype(np.float32)
        mask = (mask > 0).astype(np.float32)

        if self.augmentation:
            cube, mask = self.augmentation(cube, mask)

        return (
            torch.from_numpy(cube).float(),
            torch.from_numpy(mask).unsqueeze(0).float(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factories
# ─────────────────────────────────────────────────────────────────────────────

def build_rgb_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    root = Path(cfg["data_root"])
    img_size = cfg.get("img_size", 320)
    bs       = cfg.get("batch_size", 4)
    nw       = cfg.get("num_workers", 4)

    def _paths(split, kind):
        return sorted((root / split / kind).glob("*"))

    train_ds = RGBDataset(
        _paths("train", "img"), _paths("train", "labels"),
        transform=get_rgb_train_transforms(img_size),
    )
    val_ds   = RGBDataset(
        _paths("val", "img"), _paths("val", "labels"),
        transform=get_rgb_val_transforms(img_size),
    )
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    return train_loader, val_loader


def build_segformer_dataloaders(cfg: dict, processor: AutoImageProcessor) -> Tuple[DataLoader, DataLoader]:
    root     = Path(cfg["data_root"])
    img_size = cfg.get("img_size", 512)
    bs       = cfg.get("batch_size", 4)
    nw       = cfg.get("num_workers", 4)

    def _paths(split, kind):
        return sorted((root / split / kind).glob("*"))

    train_ds = SegFormerDataset(
        _paths("train", "img"), _paths("train", "labels"), processor,
        transform=get_segformer_train_transforms(img_size),
    )
    val_ds   = SegFormerDataset(
        _paths("val", "img"), _paths("val", "labels"), processor,
        transform=get_segformer_val_transforms(img_size),
    )
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    return train_loader, val_loader


def build_hsi_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, int]:
    root = Path(cfg["preprocessed_dir"])
    bs   = cfg.get("batch_size", 4)
    nw   = cfg.get("num_workers", 4)

    def _paths(split, kind):
        return sorted((root / split / kind).glob("*.tif"))

    # Infer n_bands from first file
    sample = _paths("train", "img")[0]
    n_bands = tifffile.imread(str(sample)).shape[0]

    from neon_tree_crown.data.hsi_augmentation import HSIAugmentation
    train_aug = HSIAugmentation(is_train=True)
    val_aug   = HSIAugmentation(is_train=False)

    train_ds = HSIDataset(_paths("train", "img"), _paths("train", "labels"), train_aug)
    val_ds   = HSIDataset(_paths("val",   "img"), _paths("val",   "labels"), val_aug)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    return train_loader, val_loader, n_bands
