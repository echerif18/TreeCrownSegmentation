from __future__ import annotations

import io
from pathlib import Path
import subprocess

import numpy as np
import streamlit as st
from PIL import Image
import tifffile
import torch

from inference import list_model_candidates, load_checkpoint, predict_mask, preprocess_rgb, resize_mask
from rag_report import CHROMA_DIR, generate_report


st.set_page_config(page_title="Tree Crown Segmentation", layout="wide")
st.title("Tree Crown Segmentation (Attention U-Net · ViT U-Net · SegFormer-B2 · HSI 3D CNN · HSI ViT)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.caption(f"Device: {device}")

project_root = Path(__file__).resolve().parents[1]
models_dir = project_root / "app" / "models"
models_dir.mkdir(parents=True, exist_ok=True)

# Recursively finds *.pth / *.pt / *.ckpt AND HuggingFace folders (config.json) at any depth
available = list_model_candidates(models_dir)
default_ckpt = str(available[0]) if available else ""

with st.sidebar:
    st.header("Model")
    selected = st.selectbox(
        "Select model from app/models/",
        options=[""] + [str(p) for p in available],
        format_func=lambda p: Path(p).relative_to(models_dir).as_posix() if p and Path(p).is_relative_to(models_dir) else (Path(p).name if p else ""),
        index=1 if available else 0,
    )
    ckpt_input = st.text_input(
        "Checkpoint path (optional override)",
        value=selected if selected else default_ckpt,
    )
    threshold = st.slider("Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    img_size = st.number_input("Model input size", min_value=64, max_value=1024, value=320, step=32)
    st.caption("Tip: for non-AttentionUNet checkpoints, provide sidecar metadata JSON in app/models/ with model_type, e.g. {\"model_type\": \"vit_rgb\"}.")

    st.subheader("RAG Report")
    use_rag = st.checkbox("Use RAG", value=False)
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
        ollama_models = [line.split()[0] for line in r.stdout.strip().split("\n")[1:] if line.strip()]
    except Exception:
        ollama_models = []
    ollama_model = st.selectbox("Ollama model", options=ollama_models or ["mistral"])
    k_docs = st.slider("Retrieved chunks (k)", min_value=3, max_value=10, value=5, step=1)
    if CHROMA_DIR.exists():
        st.success("RAG index found")
    else:
        st.warning("RAG index missing. Build with: python app/build_vectorstore.py")

if not ckpt_input:
    st.warning("Provide a checkpoint path.")
    st.stop()

ckpt_path = Path(ckpt_input).expanduser()
if not ckpt_path.is_absolute():
    ckpt_path = (project_root / ckpt_path).resolve()

if not ckpt_path.exists():
    st.error(f"Checkpoint not found: {ckpt_path}")
    st.stop()

@st.cache_resource
def _cached_model(path: str):
    return load_checkpoint(Path(path), device)

model, base_channels, metadata = _cached_model(str(ckpt_path))
st.sidebar.write(f"Base channels: {base_channels}")
if metadata:
    st.sidebar.json(metadata)

uploaded = st.file_uploader("Upload RGB image (.tif/.tiff/.png/.jpg/.jpeg)", type=["tif", "tiff", "png", "jpg", "jpeg"])
if uploaded is None:
    st.info("Upload one image to run inference.")
    st.stop()

tmp_path = project_root / ".tmp_uploaded_image"
tmp_path.write_bytes(uploaded.getvalue())

is_hsi_model = metadata.get("model_type", "") in {"hsi_3dcnn", "hsi_3d_unet", "unet3d_hsi", "vit_hsi", "hsi_vit_unet"}

if is_hsi_model:
    arr = tifffile.imread(tmp_path)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    x = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
    b_r, b_g, b_b = min(120, arr.shape[0]-1), min(60, arr.shape[0]-1), min(20, arr.shape[0]-1)
    raw = np.stack([arr[b_r], arr[b_g], arr[b_b]], axis=-1)
    lo, hi = np.percentile(raw, (2, 98))
    raw = np.clip((raw - lo) / (hi - lo + 1e-6), 0, 1)
    raw = (raw * 255).astype(np.uint8)
else:
    x, raw = preprocess_rgb(tmp_path, img_size=int(img_size))

prob, pred = predict_mask(model, x, device=device, threshold=float(threshold))
pred_full = resize_mask(pred, shape_hw=(raw.shape[0], raw.shape[1]))

overlay = raw.copy()
overlay[pred_full == 1] = (0.6 * overlay[pred_full == 1] + 0.4 * np.array([255, 50, 0])).astype(np.uint8)

col1, col2, col3 = st.columns(3)
col1.image(raw, caption="Input", use_column_width=True)
col2.image((pred_full * 255).astype(np.uint8), caption="Binary mask", use_column_width=True)
col3.image(overlay, caption="Overlay", use_column_width=True)

tree_cover_pct = float(pred_full.mean() * 100.0)
st.metric("Tree cover %", f"{tree_cover_pct:.2f}")

with st.expander("Ecological report"):
    location = st.text_input("Location (optional)", value="")
    year = st.text_input("Year (optional)", value="")
    compare_enabled = st.checkbox("Compare with previous cover", value=False)
    compare_pct = None
    if compare_enabled:
        compare_pct = st.number_input("Previous tree cover %", min_value=0.0, max_value=100.0, value=0.0)
    if st.button("Generate report"):
        report = generate_report(
            tree_cover_pct=tree_cover_pct,
            location=location or None,
            year=year or None,
            compare_pct=compare_pct,
            ollama_model=ollama_model,
            use_rag=use_rag,
            k=k_docs,
        )
        st.markdown(report)

mask_img = Image.fromarray((pred_full * 255).astype(np.uint8))
buf = io.BytesIO()
mask_img.save(buf, format="PNG")
buf.seek(0)
st.download_button("Download mask (PNG)", data=buf, file_name="pred_mask.png", mime="image/png")

try:
    tmp_path.unlink(missing_ok=True)
except Exception:
    pass
