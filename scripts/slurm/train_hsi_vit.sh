#!/bin/bash
#SBATCH --job-name=neon-hsi-vit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hsi_vit_%j.out
#SBATCH --error=logs/hsi_vit_%j.err

source "$(dirname "$0")/common.sh"
activate_project_env
print_runtime_info

run_in_project_env neon-train-vit-hsi \
  --preprocessed-dir data/hsi_preprocessed \
  --epochs           100 \
  --batch-size       8 \
  --num-workers      8

echo "HSI ViT U-Net training complete."
