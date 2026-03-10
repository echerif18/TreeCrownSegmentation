"""
Centralised config loading.  Reads YAML, then overrides with env-vars / CLI.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


# ─────────────────────────────────────────────────────────────────────────────
# Data config
# ─────────────────────────────────────────────────────────────────────────────
class DataConfig(BaseSettings):
    raw_root:           Path = Path("data/raw")
    annotations_dir:    Path = Path("data/raw/annotations")
    rgb_patches_dir:    Path = Path("data/rgb_patches")
    hsi_patches_dir:    Path = Path("data/hsi_patches")
    splits_rgb:         Path = Path("data/splits/rgb_binary")
    splits_hsi:         Path = Path("data/splits/hsi_binary")
    preprocessed_hsi:   Path = Path("data/hsi_preprocessed")

    patch_size:     int   = 320
    stride:         int   = 160
    hsi_patch:      int   = 32
    hsi_stride:     int   = 16
    val_fraction:   float = 0.15
    test_fraction:  float = 0.10


class HsiConfig(BaseSettings):
    n_bands_raw:  int   = 426
    wl_min:       float = 400.0
    wl_max:       float = 2450.0
    water_bands:  List[Tuple[float, float]] = [
        (1340, 1445), (1790, 1955), (2450, 2510)
    ]
    sg_window:    int   = 11
    sg_polyorder: int   = 3


class TrainingConfig(BaseSettings):
    seed:          int   = 42
    epochs:        int   = 100
    batch_size:    int   = 4
    num_workers:   int   = 4
    weight_decay:  float = 0.01
    clip_grad:     float = 1.0
    patience:      int   = 15

    focal_gamma:   float = 2.0
    pos_weight:    float = 2.0
    dice_weight:   float = 0.5

    lr:            float = 3e-4
    warmup_epochs: int   = 5


class WandbConfig(BaseSettings):
    project: str  = "neon-tree-crown"
    entity:  Optional[str] = None
    offline: bool = False


class OutputConfig(BaseSettings):
    runs_dir: Path = Path("runs")

    def run_dir(self, label: str) -> Path:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.runs_dir / f"{ts}_{label}" / "checkpoints"


# ─────────────────────────────────────────────────────────────────────────────
# Master config
# ─────────────────────────────────────────────────────────────────────────────
class Config(BaseSettings):
    data:     DataConfig     = Field(default_factory=DataConfig)
    hsi:      HsiConfig      = Field(default_factory=HsiConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    wandb:    WandbConfig    = Field(default_factory=WandbConfig)
    output:   OutputConfig   = Field(default_factory=OutputConfig)


def load_config(yaml_path: Optional[str | Path] = None, **overrides: Any) -> Config:
    """
    Load config from YAML (optional) then apply **overrides.
    
    Usage:
        cfg = load_config("configs/base.yaml", training__lr=1e-4)
    """
    raw: dict[str, Any] = {}

    if yaml_path is not None:
        with open(yaml_path) as f:
            raw = yaml.safe_load(f) or {}

    # Deep-merge overrides  (double-underscore → nested)
    for k, v in overrides.items():
        keys = k.split("__")
        d = raw
        for part in keys[:-1]:
            d = d.setdefault(part, {})
        d[keys[-1]] = v

    return Config(**raw)
