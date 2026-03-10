"""
cleaner.py
──────────
Remove empty/near-empty patches (no tree pixels in label) from the
RGB and HSI patch directories.

CLI:
    neon-clean --rgb-dir data/rgb_patches --hsi-dir data/hsi_patches
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import click
import numpy as np
import rasterio
from loguru import logger


def find_empty_patches(label_dir: Path, min_tree_pixels: int = 1) -> List[str]:
    """Return filenames of label files with fewer than min_tree_pixels foreground pixels."""
    empty = []
    for p in sorted(label_dir.glob("*.tif")):
        with rasterio.open(p) as src:
            mask = src.read(1)
        if int((mask > 0).sum()) < min_tree_pixels:
            empty.append(p.name)
    return empty


def remove_patches(patch_root: Path, fnames: List[str], dry_run: bool = False) -> int:
    """Remove img + label files for each fname in fnames."""
    removed = 0
    for fname in fnames:
        for sub in ["img", "labels"]:
            p = patch_root / sub / fname
            if p.exists():
                if not dry_run:
                    p.unlink()
                removed += 1
    return removed


def binarize_labels(label_dir: Path):
    """Ensure all labels are strictly {0, 1} uint8 (in-place)."""
    for p in sorted(label_dir.glob("*.tif")):
        with rasterio.open(p) as src:
            mask = src.read(1)
            meta = src.meta.copy()
        binary = (mask > 0).astype(np.uint8)
        meta.update(dtype="uint8", count=1, compress="lzw")
        with rasterio.open(p, "w", **meta) as dst:
            dst.write(binary, 1)


@click.command()
@click.option("--rgb-dir",   default="data/rgb_patches", show_default=True, type=click.Path())
@click.option("--hsi-dir",   default="data/hsi_patches", show_default=True, type=click.Path())
@click.option("--min-pixels", default=10, show_default=True, type=int,
              help="Minimum foreground pixels to keep a patch.")
@click.option("--dry-run",   is_flag=True, help="Show what would be removed without deleting.")
@click.option("--binarize",  is_flag=True, default=True, show_default=True,
              help="Convert all labels to strict {0,1} uint8 in-place.")
def main(rgb_dir, hsi_dir, min_pixels, dry_run, binarize):
    """Clean patch directories: remove empties, optionally binarize labels."""

    for name, patch_root in [("RGB", Path(rgb_dir)), ("HSI", Path(hsi_dir))]:
        lbl_dir = patch_root / "labels"
        if not lbl_dir.exists():
            logger.warning(f"{name} label dir not found: {lbl_dir}")
            continue

        if binarize:
            logger.info(f"[{name}] Binarizing labels …")
            binarize_labels(lbl_dir)

        empties = find_empty_patches(lbl_dir, min_tree_pixels=min_pixels)
        logger.info(f"[{name}] Empty patches: {len(empties)}")

        removed = remove_patches(patch_root, empties, dry_run=dry_run)
        verb    = "Would remove" if dry_run else "Removed"
        logger.info(f"[{name}] {verb} {removed} files.")

    logger.success("Cleaning complete.")


if __name__ == "__main__":
    main()
