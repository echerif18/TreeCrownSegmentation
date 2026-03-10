#!/bin/bash
#SBATCH --job-name=neon-segformer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/segformer_%j.out
#SBATCH --error=logs/segformer_%j.err

# ── Usage ─────────────────────────────────────────────────────────────────────
# sbatch scripts/slurm/train_segformer.sh
# ─────────────────────────────────────────────────────────────────────────────

source "$(dirname "$0")/common.sh"
activate_project_env
print_runtime_info

run_in_project_env neon-train-segformer \
  --data-root    data/splits/rgb_binary \
  --epochs-p1    15 \
  --epochs-p2    20 \
  --epochs-p3    25 \
  --lr-head      3e-4 \
  --lr-b3        3e-5 \
  --lr-b2        1e-5 \
  --head-dropout 0.3 \
  --batch-size   4 \
  --num-workers  8 \
  --patience     12

echo "SegFormer training complete."
