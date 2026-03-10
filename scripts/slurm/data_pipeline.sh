#!/bin/bash
#SBATCH --job-name=neon-data
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/data_pipeline_%j.out
#SBATCH --error=logs/data_pipeline_%j.err

# ── Usage ─────────────────────────────────────────────────────────────────────
# sbatch scripts/slurm/data_pipeline.sh
# Runs: patch extraction → cleaning → train/val/test split → HSI preprocessing
# ─────────────────────────────────────────────────────────────────────────────

source "$(dirname "$0")/common.sh"
activate_project_env
print_runtime_info

echo "=== [1/4] Patch extraction ==="
run_in_project_env neon-patch \
  --data-root  data/raw \
  --out-rgb    data/rgb_patches \
  --out-hsi    data/hsi_patches \
  --patch-size 320 \
  --stride     160 \
  --hsi-patch  32  \
  --hsi-stride 16

echo "=== [2/4] Cleaning (remove empties, binarise labels) ==="
run_in_project_env neon-clean \
  --rgb-dir    data/rgb_patches \
  --hsi-dir    data/hsi_patches \
  --min-pixels 10 \
  --binarize

echo "=== [3/4] Train / val / test split ==="
run_in_project_env neon-split \
  --rgb-dir   data/rgb_patches \
  --hsi-dir   data/hsi_patches \
  --out-rgb   data/splits/rgb_binary \
  --out-hsi   data/splits/hsi_binary \
  --val-frac  0.15 \
  --test-frac 0.10 \
  --seed      42

echo "=== [4/4] HSI preprocessing ==="
run_in_project_env neon-preprocess \
  --splits-dir data/splits/hsi_binary \
  --out-dir    data/hsi_preprocessed

echo "Data pipeline complete."
