# 🌲 NEON Tree Crown Segmentation

Binary segmentation of tree crowns from NEON AOP imagery using RGB and Hyperspectral (HSI) modalities.

## Models

| Model | Modality | Architecture |
|---|---|---|
| Attention U-Net | RGB | CNN encoder-decoder + attention gates |
| RGB ViT U-Net | RGB | Vision Transformer encoder + CNN decoder |
| **SegFormer-B2** | RGB | HuggingFace pretrained + progressive unfreezing |
| HSI 3D CNN U-Net | Hyperspectral | 3D spectral-spatial convolutions |
| HSI ViT U-Net | Hyperspectral | SpectralProjection + ViT U-Net |

---

## 1. Environment Setup

### Prerequisites
- Python 3.10–3.12
- [Poetry](https://python-poetry.org/docs/#installation) ≥ 1.7

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Install the project

```bash
git clone <your-repo>
cd neon_tree_crown

# Install all dependencies and the package in editable mode
poetry install

# Activate the virtual environment
poetry shell
```

> **On HPC/SLURM:** If the cluster has a GPU node with CUDA, install the CUDA-enabled torch instead:
> ```bash
> poetry run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### Environment variables (optional)

Copy `.env.example` to `.env` and fill in your WandB key:

```bash
cp .env.example .env
# Edit .env and set WANDB_API_KEY=your_key_here
```

---

## 2. Data Preparation

### Expected raw data layout

```
data/raw/
├── rgb/            ← NEON RGB GeoTIFFs  (3-band, any resolution)
├── hsi/            ← NEON HSI GeoTIFFs  (426-band hyperspectral)
└── annotations/    ← VOC-format XML bounding-box files (one per tile)
```

### Step 1 — Extract patches

```bash
neon-patch \
  --data-root  data/raw \
  --out-rgb    data/rgb_patches \
  --out-hsi    data/hsi_patches \
  --patch-size 320 \
  --stride     160 \
  --hsi-patch  32  \
  --hsi-stride 16
```

This will:
- Parse each XML → binary mask (tree=1 / background=0)
- Slide a window over each RGB tile → save 320×320 GeoTIFF patches
- Downsample labels to 1 m/px HSI grid
- Slide a window over each HSI tile → save 32×32 GeoTIFF patches

### Step 2 — Clean (remove empty patches)

```bash
neon-clean \
  --rgb-dir    data/rgb_patches \
  --hsi-dir    data/hsi_patches \
  --min-pixels 10 \
  --binarize
```

### Step 3 — Train / Val / Test split

```bash
neon-split \
  --rgb-dir   data/rgb_patches \
  --hsi-dir   data/hsi_patches \
  --out-rgb   data/splits/rgb_binary \
  --out-hsi   data/splits/hsi_binary \
  --val-frac  0.15 \
  --test-frac 0.10 \
  --seed      42
```

### Step 4 — Preprocess HSI cubes

```bash
neon-preprocess \
  --splits-dir data/splits/hsi_binary \
  --out-dir    data/hsi_preprocessed
```

This removes water-vapour bands and applies Savitzky-Golay spectral smoothing.

**Or run the whole data pipeline in one SLURM job:**

```bash
mkdir -p logs
sbatch scripts/slurm/data_pipeline.sh
```

---

## 3. Training

### Train each model individually

```bash
# Attention U-Net (RGB)
neon-train-unet \
  --data-root data/splits/rgb_binary \
  --epochs 100

# RGB ViT U-Net
neon-train-vit-rgb \
  --data-root data/splits/rgb_binary \
  --epochs 100

# SegFormer-B2 (recommended — best results)
neon-train-segformer \
  --data-root    data/splits/rgb_binary \
  --epochs-p1    15 \
  --epochs-p2    20 \
  --epochs-p3    25

# HSI 3D CNN U-Net
neon-train-hsi-cnn \
  --preprocessed-dir data/hsi_preprocessed \
  --epochs 100

# HSI ViT U-Net
neon-train-vit-hsi \
  --preprocessed-dir data/hsi_preprocessed \
  --epochs 100
```

### On SLURM

```bash
# All models sequentially (72h wall-time)
sbatch scripts/slurm/train_all_models.sh

# Or submit individually (runs in parallel)
sbatch scripts/slurm/train_rgb_unet.sh
sbatch scripts/slurm/train_rgb_vit.sh
sbatch scripts/slurm/train_segformer.sh
sbatch scripts/slurm/train_hsi_3dcnn.sh
sbatch scripts/slurm/train_hsi_vit.sh
```

Default SLURM scripts run with `poetry`. To use a Conda environment on HPC:

```bash
sbatch --export=ENV_MODE=conda,CONDA_ENV_NAME=myenv scripts/slurm/train_rgb_unet.sh
```

### WandB tracking

All training scripts log to WandB automatically. Use `--wandb-offline` to disable network access on HPC clusters:

```bash
neon-train-segformer --data-root ... --wandb-offline
```

---

## 4. Checkpoints

Checkpoints are saved to `runs/<timestamp>_<model_name>/`:

```
runs/
└── 20250310_142000_segformer_b2/
    └── best_model/           ← HuggingFace folder (SegFormer)
        ├── config.json
        ├── model.safetensors
        └── preprocessor_config.json

runs/
└── 20250310_120000_attn_unet/
    └── best_model.pth        ← PyTorch state dict (all other models)
```

---

## 5. Streamlit App

```bash
streamlit run app/app.py
```

Then open http://localhost:8501 in your browser.

Upload any `.tif` patch, select your model, point to the checkpoint, and click **Run inference**.

---

## 6. Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
neon_tree_crown/
├── pyproject.toml              ← Poetry config + CLI entry points
├── configs/
│   ├── base.yaml               ← Default hyperparameters
│   └── segformer.yaml          ← SegFormer-specific overrides
├── src/neon_tree_crown/
│   ├── data/
│   │   ├── patch_extractor.py  ← XML → patches
│   │   ├── cleaner.py          ← Remove empty patches
│   │   ├── splitter.py         ← Train/val/test split
│   │   ├── hsi_preprocessor.py ← Water band removal + Savitzky-Golay
│   │   ├── datasets.py         ← PyTorch Datasets + DataLoaders
│   │   └── hsi_augmentation.py ← Spectral + spatial HSI augmentations
│   ├── models/
│   │   ├── attention_unet.py   ← Attention U-Net
│   │   ├── segformer.py        ← SegFormer-B2 binary wrapper
│   │   ├── vit_unet.py         ← RGB & HSI ViT U-Net
│   │   ├── hsi_3dcnn.py        ← HSI 3D CNN U-Net
│   │   └── losses.py           ← Focal + Dice combined loss
│   ├── training/
│   │   ├── engine.py           ← Shared train/validate loop + EarlyStopping
│   │   ├── train_rgb_unet.py
│   │   ├── train_segformer.py  ← 3-phase progressive unfreezing
│   │   ├── train_rgb_vit.py
│   │   ├── train_hsi_3dcnn.py
│   │   └── train_hsi_vit.py
│   └── utils/
│       ├── config.py           ← Pydantic settings + YAML loader
│       └── metrics.py          ← IoU, Dice, F1
├── app/
│   └── app.py                  ← Streamlit inference app
├── scripts/slurm/
│   ├── data_pipeline.sh
│   ├── train_segformer.sh
│   └── train_all_models.sh
└── tests/
    └── test_models.py
```
