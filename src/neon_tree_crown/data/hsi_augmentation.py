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
    ):
        self.is_train           = is_train
        self.p_flip             = p_flip
        self.p_rot              = p_rot
        self.p_spectral_jitter  = p_spectral_jitter
        self.p_band_dropout     = p_band_dropout
        self.p_spectral_shift   = p_spectral_shift

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

        # ── Spectral ─────────────────────────────────────────────────────────
        if np.random.rand() < self.p_spectral_jitter:
            noise = np.random.normal(0, 0.01, cube.shape).astype(np.float32)
            cube  = np.clip(cube + noise, 0, 1)

        if np.random.rand() < self.p_band_dropout:
            n_drop = np.random.randint(1, max(2, cube.shape[0] // 20))
            idx    = np.random.choice(cube.shape[0], n_drop, replace=False)
            cube[idx] = 0.0

        if np.random.rand() < self.p_spectral_shift:
            shift = np.random.uniform(-0.02, 0.02)
            cube  = np.clip(cube + shift, 0, 1)

        return cube.astype(np.float32), mask.astype(np.float32)
