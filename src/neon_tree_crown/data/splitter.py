"""
splitter.py
───────────
Create reproducible train / val / test splits from patch directories.
Outputs are symlinks (or copies) in the canonical split structure:

    splits/rgb_binary/
        train/img/      train/labels/
        val/img/        val/labels/
        test/img/       test/labels/

CLI:
    neon-split --rgb-dir data/rgb_patches --out-dir data/splits/rgb_binary
"""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import click
from loguru import logger


def _split_names(names: List[str], val_frac: float, test_frac: float, seed: int = 42) -> Tuple:
    rng  = random.Random(seed)
    pool = sorted(names)
    rng.shuffle(pool)
    n     = len(pool)
    n_val = max(1, int(n * val_frac))
    n_tst = max(1, int(n * test_frac))
    return pool[n_val + n_tst:], pool[:n_val], pool[n_val:n_val + n_tst]


def _link_or_copy(src: Path, dst: Path, use_symlinks: bool = True):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if use_symlinks:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


@click.command()
@click.option("--rgb-dir",     default="data/rgb_patches",        show_default=True, type=click.Path())
@click.option("--hsi-dir",     default="data/hsi_patches",        show_default=True, type=click.Path())
@click.option("--out-rgb",     default="data/splits/rgb_binary",  show_default=True, type=click.Path())
@click.option("--out-hsi",     default="data/splits/hsi_binary",  show_default=True, type=click.Path())
@click.option("--val-frac",    default=0.15,  show_default=True, type=float)
@click.option("--test-frac",   default=0.10,  show_default=True, type=float)
@click.option("--seed",        default=42,    show_default=True, type=int)
@click.option("--no-symlinks", is_flag=True,  help="Copy files instead of symlinking.")
def main(rgb_dir, hsi_dir, out_rgb, out_hsi, val_frac, test_frac, seed, no_symlinks):
    """Split patches into train / val / test."""
    use_sym = not no_symlinks

    for modality, patch_root, out_root in [
        ("RGB", Path(rgb_dir), Path(out_rgb)),
        ("HSI", Path(hsi_dir), Path(out_hsi)),
    ]:
        img_dir = patch_root / "img"
        lbl_dir = patch_root / "labels"
        if not img_dir.exists():
            logger.warning(f"[{modality}] img dir not found, skipping.")
            continue

        names = sorted([p.name for p in img_dir.glob("*.tif")])
        if not names:
            logger.warning(f"[{modality}] No patches found in {img_dir}")
            continue

        train, val, test = _split_names(names, val_frac, test_frac, seed)
        logger.info(f"[{modality}] total={len(names)}  train={len(train)}  val={len(val)}  test={len(test)}")

        for split, split_names in [("train", train), ("val", val), ("test", test)]:
            for fname in split_names:
                _link_or_copy(img_dir / fname, out_root / split / "img" / fname, use_sym)
                _link_or_copy(lbl_dir / fname, out_root / split / "labels" / fname, use_sym)

        logger.success(f"[{modality}] Splits written to {out_root}")


if __name__ == "__main__":
    main()
