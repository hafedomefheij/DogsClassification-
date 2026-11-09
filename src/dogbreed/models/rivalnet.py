"""The complete model: backbone + head, for single-image inference and end-to-end training."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from .backbones import Backbone
from .heads import build_head


class RivalNet(nn.Module):
    def __init__(self, backbone: Backbone, head: nn.Module, class_names: Optional[List[str]] = None):
        super().__init__()
        self.backbone, self.head = backbone, head
        self.class_names = class_names

    def forward(self, x, y=None, token_mask=None, return_attn=False):
        g, tokens = self.backbone(x)
        return self.head(g, tokens, y=y, token_mask=token_mask, return_attn=return_attn)

    @classmethod
    def from_head_checkpoint(cls, ckpt_path: str | Path, pretrained_backbone: bool = True,
                             map_location="cpu") -> "RivalNet":
        """Rebuild backbone + trained head from a head checkpoint saved by train_heads."""
        ck = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        cfg = ck["config"]
        bb = Backbone(cfg["backbone"], cfg.get("img_size", 224), pretrained=pretrained_backbone)
        head = build_head(cfg["head"], bb.dim_global, bb.dim_token, ck["num_classes"])
        head.load_state_dict(ck["head_state"])
        if "backbone_state" in ck:
            bb.load_state_dict(ck["backbone_state"])
        return cls(bb, head, ck.get("class_names"))
