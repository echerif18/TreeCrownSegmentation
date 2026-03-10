"""
patch_extractor.py
──────────────────
Converts raw NEON tiles + XML bounding-box annotations into fixed-size
GeoTIFF patches for RGB and HSI modalities.

CLI:
    neon-patch --data-root data/raw --out-rgb data/rgb_patches --out-hsi data/hsi_patches
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

import click
import numpy as np
import rasterio
from loguru import logger
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pair_key(path: str | Path) -> str:
    """Derive pairing key from filename (strips _hyperspectral suffix)."""
    name = Path(path).stem
    return re.sub(r"_hyperspectral$", "", name)


def _pair_files(rgb_dir: Path, hsi_dir: Path) -> List[Tuple[Path, Path]]:
    rgb_map = {_pair_key(p): p for p in sorted(rgb_dir.glob("*.tif"))}
    hsi_map = {_pair_key(p): p for p in sorted(hsi_dir.glob("*.tif"))}
    keys    = sorted(rgb_map.keys() & hsi_map.keys())
    if not keys:
        raise RuntimeError(f"No matched RGB-HSI pairs found in {rgb_dir} / {hsi_dir}")
    logger.info(f"Matched {len(keys)} RGB-HSI pairs.")
    return [(hsi_map[k], rgb_map[k]) for k in keys]


def xml_to_binary_mask(xml_path: Path, width: int, height: int) -> np.ndarray:
    """Parse VOC-style XML and return uint8 binary mask (1 = tree crown)."""
    root    = ET.parse(xml_path).getroot()
    mask    = np.zeros((height, width), dtype=np.uint8)
    for obj in root.findall(".//object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = int(float(box.findtext("xmin")))
        ymin = int(float(box.findtext("ymin")))
        xmax = int(float(box.findtext("xmax")))
        ymax = int(float(box.findtext("ymax")))
        mask[ymin:ymax, xmin:xmax] = 1
    return mask


def save_geotiff(out_path: Path, arr: np.ndarray, ref_ds, transform, compress: str = "lzw"):
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    meta = ref_ds.meta.copy()
    meta.update(
        driver    = "GTiff",
        height    = arr.shape[1],
        width     = arr.shape[2],
        count     = arr.shape[0],
        dtype     = arr.dtype.name,
        transform = transform,
        compress  = compress,
    )
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(arr)


def downsample_label_to_hsi(lbl_path: Path, hsi_path: Path, out_path: Path):
    """Reproject a high-resolution label mask to the HSI spatial grid."""
    with rasterio.open(hsi_path) as hsi, rasterio.open(lbl_path) as lbl:
        dst = np.zeros((hsi.height, hsi.width), dtype=lbl.dtypes[0])
        reproject(
            source        = rasterio.band(lbl, 1),
            destination   = dst,
            src_transform = lbl.transform,
            src_crs       = lbl.crs,
            dst_transform = hsi.transform,
            dst_crs       = hsi.crs,
            dst_width     = hsi.width,
            dst_height    = hsi.height,
            resampling    = Resampling.nearest,
        )
        meta = hsi.meta.copy()
        meta.update(count=1, dtype=dst.dtype.name, compress="lzw")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **meta) as f:
            f.write(dst, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Core patching functions
# ─────────────────────────────────────────────────────────────────────────────

def patch_rgb_and_label(
    rgb_path:   Path,
    lbl_path:   Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    patch_size: int = 320,
    stride:     int = 160,
) -> int:
    """Slide a window over RGB + label, save patches as GeoTIFFs."""
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    stem  = rgb_path.stem
    count = 0

    with rasterio.open(rgb_path) as rgb_ds, rasterio.open(lbl_path) as lbl_ds:
        for row in range(0, rgb_ds.height - patch_size + 1, stride):
            for col in range(0, rgb_ds.width - patch_size + 1, stride):
                win         = Window(col, row, patch_size, patch_size)
                img_patch   = rgb_ds.read(window=win)
                lbl_patch   = lbl_ds.read(1, window=win)
                transform   = rgb_ds.window_transform(win)

                # Skip near-empty patches
                if lbl_patch.sum() == 0:
                    continue

                out_name = f"{stem}_patch{count}.tif"
                save_geotiff(out_img_dir / out_name, img_patch, rgb_ds, transform)
                save_geotiff(
                    out_lbl_dir / out_name,
                    lbl_patch.astype(np.uint8),
                    lbl_ds,
                    rasterio.open(lbl_path).window_transform(win),
                )
                count += 1

    return count


def patch_hsi_and_label(
    hsi_path:    Path,
    lbl_1m_path: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    patch_size:  int = 32,
    stride:      int = 16,
) -> int:
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    stem  = hsi_path.stem
    count = 0

    with rasterio.open(hsi_path) as hsi_ds, rasterio.open(lbl_1m_path) as lbl_ds:
        for row in range(0, hsi_ds.height - patch_size + 1, stride):
            for col in range(0, hsi_ds.width - patch_size + 1, stride):
                win       = Window(col, row, patch_size, patch_size)
                hsi_patch = hsi_ds.read(window=win)
                lbl_patch = lbl_ds.read(1, window=win)

                if lbl_patch.sum() == 0:
                    continue

                out_name = f"{stem}_patch{count}.tif"
                transform = hsi_ds.window_transform(win)
                save_geotiff(out_img_dir / out_name, hsi_patch, hsi_ds, transform)
                save_geotiff(
                    out_lbl_dir / out_name,
                    (lbl_patch > 0).astype(np.uint8),
                    lbl_ds,
                    lbl_ds.window_transform(win),
                )
                count += 1

    return count


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--data-root",    required=True, type=click.Path(), help="Root containing rgb/, hsi/, annotations/")
@click.option("--out-rgb",      default="data/rgb_patches",  show_default=True, type=click.Path())
@click.option("--out-hsi",      default="data/hsi_patches",  show_default=True, type=click.Path())
@click.option("--patch-size",   default=320,   show_default=True, type=int)
@click.option("--stride",       default=160,   show_default=True, type=int)
@click.option("--hsi-patch",    default=32,    show_default=True, type=int)
@click.option("--hsi-stride",   default=16,    show_default=True, type=int)
def main(data_root, out_rgb, out_hsi, patch_size, stride, hsi_patch, hsi_stride):
    """Extract RGB and HSI patches from raw NEON tiles."""
    root         = Path(data_root)
    rgb_dir      = root / "rgb"
    hsi_dir      = root / "hsi"
    ann_dir      = root / "annotations"
    tmp_lbl_dir  = root / "annotations" / "downsampled_1m"

    pairs = _pair_files(rgb_dir, hsi_dir)
    total_rgb = total_hsi = 0

    for hsi_path, rgb_path in pairs:
        base     = rgb_path.stem
        xml_path = ann_dir / f"{base}.xml"
        lbl_path = ann_dir / f"{base}.tif"

        # 1. XML → georeferenced binary label (RGB resolution)
        if xml_path.exists() and not lbl_path.exists():
            with rasterio.open(rgb_path) as ref:
                mask = xml_to_binary_mask(xml_path, ref.width, ref.height)
                meta = ref.meta.copy()
                meta.update(count=1, dtype="uint8", compress="lzw")
                with rasterio.open(lbl_path, "w", **meta) as dst:
                    dst.write(mask, 1)
            logger.info(f"  Saved label TIF: {lbl_path.name}")

        if not lbl_path.exists():
            logger.warning(f"No annotation for {base}, skipping.")
            continue

        # 2. RGB patches
        n = patch_rgb_and_label(
            rgb_path, lbl_path,
            Path(out_rgb) / "img", Path(out_rgb) / "labels",
            patch_size, stride,
        )
        total_rgb += n

        # 3. Downsample label → HSI grid
        lbl_1m = tmp_lbl_dir / f"{base}_1m.tif"
        tmp_lbl_dir.mkdir(parents=True, exist_ok=True)
        if not lbl_1m.exists():
            downsample_label_to_hsi(lbl_path, hsi_path, lbl_1m)

        # 4. HSI patches
        n = patch_hsi_and_label(
            hsi_path, lbl_1m,
            Path(out_hsi) / "img", Path(out_hsi) / "labels",
            hsi_patch, hsi_stride,
        )
        total_hsi += n

    logger.success(f"Done. RGB patches: {total_rgb}  |  HSI patches: {total_hsi}")


if __name__ == "__main__":
    main()
