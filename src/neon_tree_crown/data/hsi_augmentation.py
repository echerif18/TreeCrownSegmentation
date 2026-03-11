"""HSI-specific augmentations (spatial + spectral)."""
from __future__ import annotations
import numpy as np


class HSIAugmentation:
    def __init__(
        self,
        is_train:            bool  = True,
        p_flip:              float = 0.5,
        p_rot:               float = 0.5,
        p_spectral_jitter:   float = 0.5,
        p_band_dropout:      float = 0.3,
        p_spectral_shift:    float = 0.3,
        p_spatial_dropout:   float = 0.25,
        p_gamma:             float = 0.2,
    ):
        self.is_train           = is_train
        self.p_flip             = p_flip
        self.p_rot              = p_rot
        self.p_spectral_jitter  = p_spectral_jitter
        self.p_band_dropout     = p_band_dropout
        self.p_spectral_shift   = p_spectral_shift
        self.p_spatial_dropout  = p_spatial_dropout
        self.p_gamma            = p_gamma

    def __call__(self, cube: np.ndarray, mask: np.ndarray):
        if not self.is_train:
            return cube, mask

        # ── Spatial ──────────────────────────────────────────────────────────
        if np.random.rand() < self.p_flip:
            cube = cube[:, :, ::-1].copy()
            mask = mask[:, ::-1].copy()

        if np.random.rand() < self.p_flip:
            cube = cube[:, ::-1, :].copy()
            mask = mask[::-1, :].copy()

        if np.random.rand() < self.p_rot:
            k    = np.random.randint(1, 4)
            cube = np.rot90(cube, k=k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k=k).copy()

        if np.random.rand() < self.p_spatial_dropout:
            h, w = mask.shape
            cut_h = np.random.randint(max(1, h // 12), max(2, h // 5))
            cut_w = np.random.randint(max(1, w // 12), max(2, w // 5))
            top = np.random.randint(0, max(1, h - cut_h + 1))
            left = np.random.randint(0, max(1, w - cut_w + 1))
            cube[:, top:top + cut_h, left:left + cut_w] = 0.0
            mask[top:top + cut_h, left:left + cut_w] = 0.0

        # ── Spectral ─────────────────────────────────────────────────────────
        if np.random.rand() < self.p_spectral_jitter:
            noise = np.random.normal(0, 0.015, cube.shape).astype(np.float32)
            cube  = np.clip(cube + noise, 0, 1)

        if np.random.rand() < self.p_band_dropout:
            n_drop = np.random.randint(1, max(2, cube.shape[0] // 16))
            idx    = np.random.choice(cube.shape[0], n_drop, replace=False)
            cube[idx] = 0.0

        if np.random.rand() < self.p_spectral_shift:
            shift = np.random.uniform(-0.03, 0.03)
            cube  = np.clip(cube + shift, 0, 1)

        if np.random.rand() < self.p_gamma:
            gamma = np.random.uniform(0.85, 1.15)
            cube = np.clip(np.power(np.clip(cube, 0, 1), gamma), 0, 1)

        return cube.astype(np.float32), mask.astype(np.float32)
