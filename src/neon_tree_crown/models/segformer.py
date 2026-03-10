"""
SegFormer-B2 wrapper for binary tree-crown segmentation.
Progressive unfreezing: Phase 1 → head only, Phase 2 → + block[3], Phase 3 → + block[2].
"""
from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


class SegFormerBinary(nn.Module):
    """
    SegFormer-B2 fine-tuned for binary (tree / no-tree) segmentation.

    Key design decisions:
    - ignore_mismatched_sizes=True  : replaces 150-class head with 1-class head
    - use_safetensors=True          : avoids torch.load CVE-2025-32434
    - head_dropout=0.3              : stronger regularisation vs baseline 0.1
    - Progressive unfreezing        : cumulative (each phase adds more layers)
    """

    def __init__(
        self,
        pretrained_name: str = "nvidia/segformer-b2-finetuned-ade-512-512",
        img_size:        int = 512,
        head_dropout:    float = 0.3,
    ):
        super().__init__()
        self.img_size  = img_size
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_name,
            ignore_mismatched_sizes = True,
            num_labels              = 1,
            use_safetensors         = True,
        )
        self.segformer.decode_head.dropout.p = head_dropout

    # ── Freezing helpers ──────────────────────────────────────────────────────

    def freeze_all_encoder(self):
        """Phase 1: train decode head only."""
        for p in self.segformer.segformer.parameters():
            p.requires_grad = False

    def unfreeze_block3(self):
        """Phase 2: additionally unfreeze encoder block[3]."""
        for p in self.segformer.segformer.encoder.block[-1].parameters():
            p.requires_grad = True
        for p in self.segformer.segformer.encoder.layer_norm[-1].parameters():
            p.requires_grad = True

    def unfreeze_block2(self):
        """Phase 3: additionally unfreeze encoder block[2]."""
        for p in self.segformer.segformer.encoder.block[-2].parameters():
            p.requires_grad = True
        for p in self.segformer.segformer.encoder.layer_norm[-2].parameters():
            p.requires_grad = True

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = (x.shape[2], x.shape[3])
        logits = self.segformer(pixel_values=x).logits     # (B,1,H/4,W/4)
        return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path):
        """Save via HuggingFace save_pretrained (produces a folder)."""
        self.segformer.save_pretrained(path)

    @classmethod
    def load(cls, path: str | Path, img_size: int = 512, device: str = "cpu") -> "SegFormerBinary":
        obj            = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.img_size   = img_size
        obj.segformer  = SegformerForSemanticSegmentation.from_pretrained(path)
        return obj.to(device)
