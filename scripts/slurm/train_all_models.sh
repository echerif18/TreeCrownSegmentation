#!/bin/bash
#SBATCH --job-name=neon-all-models
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=logs/train_all_%j.out
#SBATCH --error=logs/train_all_%j.err

# ── Usage ─────────────────────────────────────────────────────────────────────
# sbatch scripts/slurm/train_all_models.sh
# Trains all 5 models sequentially on a single GPU node.
# For parallel training, submit each model as a separate job.
# ─────────────────────────────────────────────────────────────────────────────

source "$(dirname "$0")/common.sh"
activate_project_env
print_runtime_info

echo "=============================="
echo " Training Attention U-Net"
echo "=============================="
run_in_project_env neon-train-unet \
  --data-root   data/splits/rgb_binary \
  --epochs      100 \
  --batch-size  4   \
  --num-workers 8

echo "=============================="
echo " Training RGB ViT U-Net"
echo "=============================="
run_in_project_env neon-train-vit-rgb \
  --data-root   data/splits/rgb_binary \
  --epochs      100 \
  --batch-size  4

echo "=============================="
echo " Training SegFormer-B2"
echo "=============================="
run_in_project_env neon-train-segformer \
  --data-root   data/splits/rgb_binary \
  --epochs-p1   15 \
  --epochs-p2   20 \
  --epochs-p3   25

echo "=============================="
echo " Training HSI 3D CNN U-Net"
echo "=============================="
run_in_project_env neon-train-hsi-cnn \
  --preprocessed-dir data/hsi_preprocessed \
  --epochs           100 \
  --batch-size       4

echo "=============================="
echo " Training HSI ViT U-Net"
echo "=============================="
run_in_project_env neon-train-vit-hsi \
  --preprocessed-dir data/hsi_preprocessed \
  --epochs           100 \
  --batch-size       4

echo "All models trained."
