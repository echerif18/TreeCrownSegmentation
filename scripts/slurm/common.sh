#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/echerif/ironhack/neon_tree_crown_v3}"
ENV_MODE="${ENV_MODE:-poetry}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-crown-clean}"

mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

if command -v module >/dev/null 2>&1; then
  module load anaconda/3 >/dev/null 2>&1 || true
fi

activate_project_env() {
  case "${ENV_MODE}" in
    poetry)
      if ! command -v poetry >/dev/null 2>&1; then
        echo "poetry not found in PATH" >&2
        exit 1
      fi
      ;;
    conda)
      if ! command -v conda >/dev/null 2>&1; then
        echo "conda not found in PATH" >&2
        exit 1
      fi
      eval "$(conda shell.bash hook)"
      conda activate "${CONDA_ENV_NAME}"
      ;;
    *)
      echo "Unsupported ENV_MODE='${ENV_MODE}'. Use 'poetry' or 'conda'." >&2
      exit 1
      ;;
  esac
}

run_in_project_env() {
  if [[ "${ENV_MODE}" == "poetry" ]]; then
    poetry run "$@"
  else
    "$@"
  fi
}

print_runtime_info() {
  echo "Project root: ${PROJECT_ROOT}"
  echo "Environment mode: ${ENV_MODE}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
  else
    echo "GPU: unavailable"
  fi
  run_in_project_env python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
}
