"""
convert_segformer_to_pth.py
────────────────────────────
Convert an existing SegFormer HuggingFace folder (produced by the old
save_pretrained-only approach) into a standard best_model.pth file that
works identically to all other neon_tree_crown model checkpoints.

Usage:
    # Convert one folder
    poetry run python scripts/convert_segformer_to_pth.py \
        --hf-dir outputs/20240101_segformer_b2/best_model

    # Convert all best_model/ folders under a search root
    poetry run python scripts/convert_segformer_to_pth.py \
        --search-root outputs/

    # Dry run (shows what would be created, writes nothing)
    poetry run python scripts/convert_segformer_to_pth.py \
        --search-root outputs/ --dry-run

Output:
    For each   <hf_dir>/          (contains config.json)
    Writes     <hf_dir>.pth       (standard checkpoint dict)

    The .pth contains:
        {
            "model_state":     state_dict,          # all weights
            "model_type":      "segformer",          # auto-detected by load_checkpoint
            "img_size":        512,
            "pretrained_name": "nvidia/segformer-b2-finetuned-ade-512-512",
            "head_dropout":    0.3,
        }
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def convert(hf_dir: Path, dry_run: bool = False) -> Path:
    """Convert one HuggingFace SegFormer folder to .pth. Returns output path."""
    if not (hf_dir / "config.json").exists():
        raise ValueError(f"{hf_dir} does not look like a HuggingFace model folder (no config.json)")

    out_path = hf_dir.parent / f"{hf_dir.name}.pth"

    # Read img_size from config if available
    import json
    config_path = hf_dir / "config.json"
    with open(config_path) as f:
        hf_config = json.load(f)
    # SegFormer configs store image_size or use default 512
    img_size = hf_config.get("image_size", 512)
    if isinstance(img_size, list):
        img_size = img_size[0]

    pretrained_name = hf_config.get(
        "_name_or_path", "nvidia/segformer-b2-finetuned-ade-512-512"
    )

    if dry_run:
        print(f"  [DRY RUN] Would convert:\n    {hf_dir}\n  → {out_path}")
        print(f"    img_size={img_size}  pretrained_name={pretrained_name}")
        return out_path

    from transformers import SegformerForSemanticSegmentation
    print(f"  Loading weights from {hf_dir} …")
    model = SegformerForSemanticSegmentation.from_pretrained(str(hf_dir))
    model.eval()

    # Try to read head_dropout from config
    head_dropout = 0.3
    if hasattr(model.config, "hidden_dropout_prob"):
        head_dropout = model.config.hidden_dropout_prob

    ckpt = {
        "model_state":     model.state_dict(),
        "model_type":      "segformer",
        "img_size":        img_size,
        "pretrained_name": pretrained_name,
        "head_dropout":    head_dropout,
    }

    torch.save(ckpt, out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  ✅ Saved → {out_path}  ({size_mb:.1f} MB)")
    return out_path


def find_hf_folders(root: Path) -> list[Path]:
    """Recursively find all HuggingFace model folders under root."""
    return sorted(p.parent for p in root.rglob("config.json")
                  if (p.parent / "config.json").exists()
                  and not (p.parent / "config.json").parent.name.startswith("."))


def main():
    parser = argparse.ArgumentParser(
        description="Convert SegFormer HuggingFace folders → best_model.pth"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--hf-dir",      type=Path,
                       help="Single HuggingFace folder to convert")
    group.add_argument("--search-root", type=Path,
                       help="Root directory to scan recursively for HF folders")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing any files")
    args = parser.parse_args()

    if args.hf_dir:
        folders = [args.hf_dir]
    else:
        folders = find_hf_folders(args.search_root)
        if not folders:
            print(f"No HuggingFace folders found under {args.search_root}")
            return
        print(f"Found {len(folders)} HuggingFace folder(s):")
        for f in folders:
            print(f"  {f}")
        print()

    converted = []
    errors    = []
    for folder in folders:
        print(f"Converting: {folder}")
        try:
            out = convert(folder, dry_run=args.dry_run)
            converted.append(out)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            errors.append((folder, e))

    print()
    if args.dry_run:
        print(f"Dry run complete. Would convert {len(converted)} folder(s).")
    else:
        print(f"Done. Converted {len(converted)} folder(s).")
        if errors:
            print(f"Errors: {len(errors)}")
        print()
        print("Copy to app/models/segformer/ and run the app:")
        for out in converted:
            print(f"  cp {out} app/models/segformer/")


if __name__ == "__main__":
    main()
