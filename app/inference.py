from __future__ import annotations

from pathlib import Path
import importlib
import json
import sys
import types

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure app always uses local project sources, not an older installed package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from neon_tree_crown.models.attention_unet import AttentionUNet
from neon_tree_crown.models.hsi_3dcnn import HSI3DUNet
from neon_tree_crown.models.segformer import SegFormerBinary
from neon_tree_crown.models.vit_unet import RGBViTUNet, HSIViTUNet


def _read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        arr = tifffile.imread(path)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[0] in {3, 4} and arr.shape[-1] not in {3, 4}:
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    return np.array(Image.open(path).convert("RGB"))


def preprocess_rgb(path: Path, img_size: int = 320) -> tuple[torch.Tensor, np.ndarray]:
    image = _read_image(path)
    raw = image.copy()
    pil = Image.fromarray(image).resize((img_size, img_size), Image.BILINEAR)
    x = np.asarray(pil).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    tensor = torch.from_numpy(x).unsqueeze(0).float()
    return tensor, raw


def infer_base_channels(state_dict: dict) -> int:
    # neon_tree_crown AttentionUNet uses enc1.block.0.weight
    key = "enc1.block.0.weight"
    if key in state_dict:
        return int(state_dict[key].shape[0])
    return 64   # neon_tree_crown default (was 32 in treeCrownTest)


def infer_vitunet_params(state_dict: dict, metadata: dict) -> dict:
    """Infer RGBViTUNet architecture params from state-dict + sidecar metadata."""
    hidden = int(metadata.get("vit_hidden", metadata.get("hidden", metadata.get("embed_dim", 384))))
    if "norm.weight" in state_dict:
        hidden = int(state_dict["norm.weight"].shape[0])
    elif "transformer.0.norm1.weight" in state_dict:
        hidden = int(state_dict["transformer.0.norm1.weight"].shape[0])

    n_layers = int(metadata.get("vit_depth", metadata.get("depth", metadata.get("n_layers", 6))))
    block_ids = []
    for k in state_dict.keys():
        if k.startswith("blocks.") or k.startswith("transformer."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_ids.append(int(parts[1]))
    if block_ids:
        n_layers = max(block_ids) + 1

    patch_size = int(metadata.get("patch_size", 16))
    if "patch_embed.proj.weight" in state_dict:
        patch_size = int(state_dict["patch_embed.proj.weight"].shape[-1])

    img_size = int(metadata.get("img_size", 320))
    if "pos_embed" in state_dict:
        n_tokens = int(state_dict["pos_embed"].shape[1])
        if n_tokens > 1:
            n_tokens -= 1  # RGBViTUNet stores a CLS token in pos_embed
        side = int(round(n_tokens ** 0.5))
        img_size = side * patch_size

    mlp_dim = int(metadata.get("vit_mlp_dim", metadata.get("mlp_dim", 1536)))
    if "transformer.0.mlp.0.weight" in state_dict:
        mlp_dim = int(state_dict["transformer.0.mlp.0.weight"].shape[0])
    elif "blocks.0.mlp.0.weight" in state_dict:
        mlp_dim = int(state_dict["blocks.0.mlp.0.weight"].shape[0])

    return {
        "img_size": img_size,
        "patch_size": patch_size,
        "hidden": hidden,
        "depth": n_layers,
        "heads": int(metadata.get("vit_heads", metadata.get("heads", metadata.get("n_heads", 8)))),
        "mlp_dim": mlp_dim,
        "drop": float(metadata.get("dropout", metadata.get("drop", 0.1))),
    }


def infer_hsi_vit_params(state_dict: dict, metadata: dict) -> dict:
    """Infer HSIViTUNet architecture params from state-dict + sidecar metadata."""
    hidden = int(metadata.get("vit_hidden", metadata.get("hidden", 256)))
    if "vit_rgb.pos_embed" in state_dict:
        hidden = int(state_dict["vit_rgb.pos_embed"].shape[-1])
    elif "pos_embed" in state_dict:
        hidden = int(state_dict["pos_embed"].shape[-1])

    depth = int(metadata.get("vit_depth", metadata.get("depth", 4)))
    block_ids = []
    for k in state_dict.keys():
        if k.startswith("vit_rgb.transformer.") or k.startswith("blocks."):
            parts = k.split(".")
            idx_pos = 2 if k.startswith("vit_rgb.transformer.") else 1
            if len(parts) > idx_pos and parts[idx_pos].isdigit():
                block_ids.append(int(parts[idx_pos]))
    if block_ids:
        depth = max(block_ids) + 1

    patch_size = int(metadata.get("patch_size", 4))
    if "vit_rgb.patch_embed.proj.weight" in state_dict:
        patch_size = int(state_dict["vit_rgb.patch_embed.proj.weight"].shape[-1])
    elif "patch_embed.proj.weight" in state_dict:
        patch_size = int(state_dict["patch_embed.proj.weight"].shape[-1])

    img_size = int(metadata.get("img_size", 32))
    mlp_dim  = int(metadata.get("vit_mlp_dim", metadata.get("mlp_dim", 1024)))
    if "vit_rgb.transformer.0.mlp.0.weight" in state_dict:
        mlp_dim = int(state_dict["vit_rgb.transformer.0.mlp.0.weight"].shape[0])

    spectral_out = int(metadata.get("spectral_out", 64))
    if "spectral.proj.3.weight" in state_dict:
        spectral_out = int(state_dict["spectral.proj.3.weight"].shape[0])
    elif "spectral.proj.0.weight" in state_dict:
        spectral_out = int(state_dict["spectral.proj.0.weight"].shape[0])

    return {
        "n_bands": int(metadata.get("n_bands", 356)),
        "spectral_out": spectral_out,
        "img_size": img_size,
        "patch_size": patch_size,
        "depth": depth,
        "heads": int(metadata.get("vit_heads", metadata.get("heads", metadata.get("n_heads", 8)))),
        "hidden": hidden,
        "mlp_dim": mlp_dim,
        "drop": float(metadata.get("dropout", metadata.get("drop", 0.1))),
    }


def _extract_state(ckpt_obj):
    if isinstance(ckpt_obj, torch.nn.Module):
        state = ckpt_obj.state_dict()
        meta = {"model_meta": {"model_class_loaded": ckpt_obj.__class__.__name__}}
        return state, meta
    if isinstance(ckpt_obj, dict) and "model_state" in ckpt_obj:
        return ckpt_obj["model_state"], ckpt_obj   # neon_tree_crown checkpoint format
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        return ckpt_obj["state_dict"], ckpt_obj
    return ckpt_obj, {}


def _load_sidecar_metadata(ckpt_path: Path) -> dict:
    sidecar = ckpt_path.with_suffix(".json")
    if sidecar.exists():
        with sidecar.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _build_model_from_metadata(metadata: dict, state_dict: dict, device: torch.device):
    # Option A: built-in short names
    model_type = metadata.get("model_type", "").strip().lower()

    if model_type in {"attention_unet", "att_unet", "attn_unet"}:
        base_channels = int(metadata.get("base_channels", infer_base_channels(state_dict)))
        model = AttentionUNet(in_channels=3, out_channels=1, base_channels=base_channels).to(device)
        return model, base_channels, "attn_unet"

    if model_type in {"vit_rgb", "rgb_vit_unet"}:
        vit = infer_vitunet_params(state_dict, metadata)
        model = RGBViTUNet(**vit).to(device)
        return model, int(vit["hidden"]), "vit_rgb"

    if model_type in {"segformer_b2", "segformer"}:
        # Load directly from the HuggingFace folder — no wrapper, no state_dict remapping.
        # metadata["hf_path"] is set by load_checkpoint when it detects a HF folder.
        hf_path = metadata.get("hf_path") or metadata.get("pretrained_name",
                  "nvidia/segformer-b2-finetuned-ade-512-512")
        from transformers import SegformerForSemanticSegmentation, AutoImageProcessor
        model     = SegformerForSemanticSegmentation.from_pretrained(hf_path).to(device)
        processor = AutoImageProcessor.from_pretrained(hf_path)
        metadata["_processor"] = processor
        metadata["img_size"]   = int(metadata.get("img_size", 512))
        return model, -1, "segformer"

    if model_type in {"hsi_3dcnn", "hsi_3d_unet", "unet3d_hsi"}:
        model = HSI3DUNet(
            n_bands       = int(metadata.get("n_bands", 356)),
            base_channels = int(metadata.get("base_channels", 8)),
        ).to(device)
        return model, int(metadata.get("base_channels", 8)), "hsi_3dcnn"

    if model_type in {"vit_hsi", "hsi_vit_unet"}:
        hsi = infer_hsi_vit_params(state_dict, metadata)
        model = HSIViTUNet(**hsi).to(device)
        return model, int(hsi["hidden"]), "vit_hsi"

    # Option B: fully qualified class path
    model_module = metadata.get("model_module")
    model_class  = metadata.get("model_class")
    model_kwargs = metadata.get("model_kwargs", {})
    if model_module and model_class:
        module = importlib.import_module(model_module)
        cls    = getattr(module, model_class)
        model  = cls(**model_kwargs).to(device)
        inferred_type = metadata.get("model_type", model_class)
        return model, int(model_kwargs.get("base_channels", -1)), inferred_type

    # Heuristic auto-detection by key patterns.
    keys = set(state_dict.keys())

    if any(k.startswith("spectral.proj.") or k.startswith("vit_rgb.patch_embed.") for k in keys):
        # HSIViTUNet has a spectral projection layer
        hsi = infer_hsi_vit_params(state_dict, metadata)
        model = HSIViTUNet(**hsi).to(device)
        return model, int(hsi["hidden"]), "vit_hsi"

    if "spectral_collapse.0.output_size" in keys or any(v.ndim == 5 for v in state_dict.values() if hasattr(v, "ndim")):
        model = HSI3DUNet(
            n_bands       = int(metadata.get("n_bands", 356)),
            base_channels = int(metadata.get("base_channels", 8)),
        ).to(device)
        return model, int(metadata.get("base_channels", 8)), "hsi_3dcnn"

    if any("segformer.encoder.patch_embeddings" in k for k in keys):
        model = SegFormerBinary(pretrained_name="nvidia/segformer-b2-finetuned-ade-512-512").to(device)
        return model, -1, "segformer"

    if any(k.startswith("patch_embed.proj.") for k in keys) and "pos_embed" in keys:
        vit = infer_vitunet_params(state_dict, metadata)
        model = RGBViTUNet(**vit).to(device)
        return model, int(vit["hidden"]), "vit_rgb"

    # Fallback: Attention U-Net auto-detection.
    base_channels = infer_base_channels(state_dict)
    model = AttentionUNet(in_channels=3, out_channels=1, base_channels=base_channels).to(device)
    return model, base_channels, "attn_unet"


def _remap_state_keys_for_model(model: torch.nn.Module, state: dict) -> dict:
    """Make legacy checkpoint keys compatible with current wrappers."""
    if not isinstance(state, dict):
        return state

    # SegFormer: normalise key prefixes to what SegFormerBinary expects.
    # SegFormerBinary stores the HF model as self.segformer, so the expected
    # key structure is:
    #   segformer.segformer.encoder.*   (encoder blocks)
    #   segformer.decode_head.*         (decode head)
    #
    # Checkpoints saved before the wrapper was finalised may use:
    #   model.encoder.*  / model.decode_head.*   (raw HF state_dict)
    #   segformer.segformer.* / segformer.decode_head.*  (wrapped correctly)
    if isinstance(model, SegFormerBinary):
        remapped = {}
        for k, v in state.items():
            nk = k
            # Already correct — leave alone
            if nk.startswith("segformer.segformer.") or nk.startswith("segformer.decode_head."):
                pass
            # Raw HF keys: "model.encoder.*" → "segformer.segformer.encoder.*"
            elif nk.startswith("model.encoder.") or nk.startswith("model.segformer.encoder."):
                nk = "segformer.segformer." + nk[len("model."):]
            # Raw HF decode_head: "model.decode_head.*" → "segformer.decode_head.*"
            elif nk.startswith("model.decode_head."):
                nk = "segformer." + nk[len("model."):]
            # Bare HF keys without "model." prefix
            elif nk.startswith("encoder."):
                nk = "segformer.segformer." + nk
            elif nk.startswith("decode_head."):
                nk = "segformer." + nk
            remapped[nk] = v
        return remapped

    return state


def list_model_candidates(models_dir: Path) -> list[Path]:
    """
    Recursively find all model checkpoints under models_dir.
    Supports:
      - *.pth / *.pt / *.ckpt  (any depth)
      - HuggingFace folders     (any depth, must contain config.json)
    Deduplicates: if a HF folder and a sibling .pth both exist for the
    same checkpoint, the folder takes priority (it's self-contained).
    """
    exts = (".pth", ".ckpt", ".pt")

    # Collect all .pth/.pt/.ckpt files recursively
    files = [p for p in models_dir.rglob("*") if p.suffix.lower() in exts]

    # Collect HuggingFace folders recursively
    hf_dirs = [p.parent for p in models_dir.rglob("config.json")]

    # Remove .pth files whose stem matches a sibling HF folder
    # e.g. best_model.pth alongside best_model/ → keep only best_model/
    hf_dir_set = set(hf_dirs)
    files = [
        f for f in files
        if f.parent / f.stem not in hf_dir_set
    ]

    return sorted(set(files + hf_dirs))


def _register_legacy_class_aliases() -> None:
    """Support checkpoints that were saved with torch.save(model) in notebooks."""
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        return
    aliases = {
        "AttentionUNet":    AttentionUNet,
        "HSI3DUNet":        HSI3DUNet,
        "SegFormerBinary":  SegFormerBinary,
        "RGBViTUNet":       RGBViTUNet,
        "HSIViTUNet":       HSIViTUNet,
    }
    for name, cls in aliases.items():
        if not hasattr(main_mod, name):
            setattr(main_mod, name, cls)


def _register_transformers_legacy_aliases() -> None:
    """Some old pickled checkpoints reference moved transformers internals."""
    if "transformers.core_model_loading" in sys.modules:
        return
    try:
        import transformers.modeling_utils as modeling_utils
    except Exception:
        return
    if not hasattr(modeling_utils, "WeightRenaming"):
        class WeightRenaming:
            def __init__(self, *args, **kwargs):
                self.args   = args
                self.kwargs = kwargs
        setattr(modeling_utils, "WeightRenaming", WeightRenaming)
    shim = types.ModuleType("transformers.core_model_loading")
    for name in dir(modeling_utils):
        try:
            setattr(shim, name, getattr(modeling_utils, name))
        except Exception:
            continue
    sys.modules["transformers.core_model_loading"] = shim


def load_checkpoint(ckpt_path: Path, device: torch.device):
    # ── SegFormer: any HuggingFace folder (config.json present) ──────────────
    # Handles: app/models/segformer/best_model/  or any nested HF folder.
    # Exactly equivalent to:
    #   model     = SegformerForSemanticSegmentation.from_pretrained(path).to(device)
    #   processor = AutoImageProcessor.from_pretrained(path)
    #   model.eval()
    if ckpt_path.is_dir() and (ckpt_path / "config.json").exists():
        from transformers import SegformerForSemanticSegmentation, AutoImageProcessor
        best_model     = SegformerForSemanticSegmentation.from_pretrained(str(ckpt_path)).to(device)
        best_processor = AutoImageProcessor.from_pretrained(str(ckpt_path))
        best_model.eval()
        return best_model, -1, {
            "model_type": "segformer",
            "hf_path":    str(ckpt_path),
            "_processor": best_processor,
            "img_size":   512,
        }

    _register_legacy_class_aliases()
    _register_transformers_legacy_aliases()

    load_errors = []
    ckpt = None
    # 1) Prefer weight-only load to avoid pickle/import issues.
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
        except Exception as e:
            load_errors.append(e)
    except Exception as e:
        load_errors.append(e)

    # 2) Fallback for full-object checkpoints.
    if ckpt is None:
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)
        except Exception as e:
            load_errors.append(e)

    if ckpt is None:
        msg = " | ".join(str(e) for e in load_errors) if load_errors else "unknown error"
        raise RuntimeError(f"Failed to load checkpoint '{ckpt_path}': {msg}")

    state, raw = _extract_state(ckpt)
    metadata = _load_sidecar_metadata(ckpt_path)
    # Allow metadata inside checkpoint too (neon_tree_crown saves cfg + model_meta).
    if isinstance(raw, dict):
        metadata = {
            **raw.get("cfg", {}),
            **raw.get("model_meta", {}),
            # promote top-level keys neon_tree_crown trainers write directly
            **{k: raw[k] for k in ("model_type", "n_bands", "base_channels",
                                    "val_iou", "epoch", "hf_path",
                                    "pretrained_name", "img_size") if k in raw},
            **metadata,
        }
    model, base_channels, inferred_type = _build_model_from_metadata(metadata, state, device)
    metadata.setdefault("model_type", inferred_type)
    state = _remap_state_keys_for_model(model, state)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, base_channels, metadata


@torch.no_grad()
def predict_mask(model, x: torch.Tensor, device: torch.device,
                 threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    x = x.to(device)
    # SegFormer: always call with pixel_values= and extract .logits
    if hasattr(model, "config") and getattr(model.config, "model_type", "") == "segformer":
        out    = model(pixel_values=x)
        logits = out.logits                            # (B, 1, H/4, W/4)
        logits = F.interpolate(logits, size=(x.shape[-2], x.shape[-1]),
                               mode="bilinear", align_corners=False)
        probs  = torch.sigmoid(logits[:, 0:1])
    else:
        probs = torch.sigmoid(model(x))
    pred    = (probs > threshold).float()
    prob_np = probs.squeeze().cpu().numpy()
    pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)
    return prob_np, pred_np


def resize_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
    t = F.interpolate(t, size=shape_hw, mode="nearest")
    return t.squeeze().numpy().astype(np.uint8)
