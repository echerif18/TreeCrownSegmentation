#!/bin/bash
#SBATCH --job-name=neon-rgb-vit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rgb_vit_%j.out
#SBATCH --error=logs/rgb_vit_%j.err

source "$(dirname "$0")/common.sh"
activate_project_env
print_runtime_info

run_in_project_env neon-train-vit-rgb \
  --data-root   data/splits/rgb_binary \
  --epochs      100 \
  --batch-size  8 \
  --num-workers 8

echo "RGB ViT U-Net training complete."
