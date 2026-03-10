#!/bin/bash
#SBATCH --job-name=neon-hsi-3dcnn
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hsi_3dcnn_%j.out
#SBATCH --error=logs/hsi_3dcnn_%j.err

source "$(dirname "$0")/common.sh"
activate_project_env
print_runtime_info

run_in_project_env neon-train-hsi-cnn \
  --preprocessed-dir data/hsi_preprocessed \
  --epochs           100 \
  --batch-size       8 \
  --num-workers      8

echo "HSI 3D CNN U-Net training complete."
