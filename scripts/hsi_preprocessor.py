"""
hsi_preprocessor.py
────────────────────
Clean raw HSI cubes:
  1. Scale NEON AOP reflectance (÷10000)
  2. Remove water-vapour & noisy bands
  3. Apply Savitzky-Golay smoothing
  4. Save preprocessed cubes (float32 tifffile format)

CLI:
    neon-preprocess --splits-dir data/splits/hsi_binary --out-dir data/hsi_preprocessed
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple

import click
import numpy as np
import tifffile
from loguru import logger
from scipy.signal import savgol_filter


# ─────────────────────────────────────────────────────────────────────────────
# Spectral helpers
# ─────────────────────────────────────────────────────────────────────────────

WATER_BANDS: List[Tuple[float, float]] = [
    (1340, 1445),
    (1790, 1955),
    (2450, 2510),
]
WL_MIN = 400.0
WL_MAX = 2450.0
N_BANDS_RAW = 426


def get_wavelengths(n_bands: int = N_BANDS_RAW) -> np.ndarray:
    return np.linspace(380, 2510, n_bands)


def get_band_mask(wl: np.ndarray) -> np.ndarray:
    """Return boolean array: True = keep this band."""
    bad = np.zeros(len(wl), dtype=bool)
    for lo, hi in WATER_BANDS:
        bad |= (wl >= lo) & (wl <= hi)
    bad |= (wl < WL_MIN) | (wl > WL_MAX)
    return ~bad


def preprocess_cube(
    cube: np.ndarray,
    band_mask: np.ndarray,
    sg_window: int = 11,
    sg_polyorder: int = 3,
) -> np.ndarray:
    """
    cube      : (B, H, W) float32  raw NEON reflectance (×10 000)
    Returns   : (B_clean, H, W) float32 cleaned reflectance [0, 1]
    """
    cube = cube / 10_000.0
    cube = np.clip(cube, 0.0, 1.0)
    cube = cube[band_mask]                            # remove bad bands
    cube = savgol_filter(cube, window_length=sg_window,
                         polyorder=sg_polyorder, axis=0)  # spectral smooth
    return cube.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main preprocessing loop
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_split(
    split_dir: Path,
    out_dir:   Path,
    n_bands_raw: int = N_BANDS_RAW,
    sg_window:   int = 11,
    sg_polyorder: int = 3,
) -> int:
    import rasterio  # keep optional at module level

    wl        = get_wavelengths(n_bands_raw)
    band_mask = get_band_mask(wl)
    n_clean   = int(band_mask.sum())
    logger.info(f"  Keeping {n_clean}/{n_bands_raw} bands after water-band removal.")

    img_in  = split_dir / "img"
    lbl_in  = split_dir / "labels"
    img_out = out_dir   / "img"
    lbl_out = out_dir   / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    files = sorted(img_in.glob("*.tif"))
    for i, src_path in enumerate(files):
        dst_path = img_out / src_path.name

        with rasterio.open(src_path) as ds:
            raw = ds.read().astype(np.float32)          # (B, H, W)

        clean = preprocess_cube(raw, band_mask, sg_window, sg_polyorder)
        tifffile.imwrite(str(dst_path), clean, photometric="minisblack")

        # Copy label unchanged
        lbl_src = lbl_in / src_path.name
        lbl_dst = lbl_out / src_path.name
        if lbl_src.exists() and not lbl_dst.exists():
            shutil.copy2(lbl_src, lbl_dst)

        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i + 1}/{len(files)} …")

    return n_clean


@click.command()
@click.option("--splits-dir",   default="data/splits/hsi_binary",  show_default=True, type=click.Path())
@click.option("--out-dir",      default="data/hsi_preprocessed",   show_default=True, type=click.Path())
@click.option("--n-bands-raw",  default=426,  show_default=True, type=int)
@click.option("--sg-window",    default=11,   show_default=True, type=int)
@click.option("--sg-polyorder", default=3,    show_default=True, type=int)
def main(splits_dir, out_dir, n_bands_raw, sg_window, sg_polyorder):
    """Preprocess HSI cubes: scale, remove water bands, Savitzky-Golay smooth."""
    root = Path(splits_dir)
    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            logger.warning(f"Split dir not found: {split_dir}")
            continue
        logger.info(f"Preprocessing [{split}] …")
        n_clean = preprocess_split(
            split_dir, Path(out_dir) / split,
            n_bands_raw, sg_window, sg_polyorder,
        )
    logger.success(f"Done. Clean band count per cube = {n_clean}")


if __name__ == "__main__":
    main()
