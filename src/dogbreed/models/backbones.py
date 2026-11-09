"""Pretrained backbones as (global descriptor, spatial tokens) extractors.

Inherited components (not contributions of this work): every backbone here is used with its
published pretrained weights. Global descriptor g:
* ViTs (DINOv2, CLIP): [CLS ; mean of patch tokens] (the DINOv2 linear-evaluation recipe);
  register tokens of the "-reg" models are dropped from the spatial tokens.
* CNNs (ResNet-50, ConvNeXt-T): global average of the last feature map.
Pretraining-data caveats (ImageNet overlaps Stanford Dogs) are recorded per entry and
reported with every result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..datasets.transforms import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class BackboneSpec:
    lib: str                     # "timm" or "torchvision"
    model_id: str
    patch: int                   # token stride in pixels
    mean: Tuple[float, ...] = IMAGENET_MEAN
    std: Tuple[float, ...] = IMAGENET_STD
    weights: Optional[str] = None
    pretraining: str = ""
    supervised_imagenet: bool = False   # trained WITH ImageNet labels (label leakage on Stanford Dogs)
    notes: str = field(default="", compare=False)


BACKBONES = {
    "dinov2_vits14_reg": BackboneSpec(
        "timm", "vit_small_patch14_reg4_dinov2.lvd142m", 14,
        pretraining="DINOv2 self-supervised, LVD-142M (no labels)"),
    "dinov2_vitb14_reg": BackboneSpec(
        "timm", "vit_base_patch14_reg4_dinov2.lvd142m", 14,
        pretraining="DINOv2 self-supervised, LVD-142M (no labels)"),
    "dinov2_vitl14_reg": BackboneSpec(
        "timm", "vit_large_patch14_reg4_dinov2.lvd142m", 14,
        pretraining="DINOv2 self-supervised, LVD-142M (no labels)"),
    "clip_vitb16": BackboneSpec(
        "timm", "vit_base_patch16_clip_224.openai", 16, CLIP_MEAN, CLIP_STD,
        pretraining="CLIP image-text contrastive, WIT-400M"),
    "resnet50": BackboneSpec(
        "torchvision", "resnet50", 32, weights="IMAGENET1K_V2",
        pretraining="supervised ImageNet-1k", supervised_imagenet=True),
    "convnext_tiny": BackboneSpec(
        "torchvision", "convnext_tiny", 32, weights="IMAGENET1K_V1",
        pretraining="supervised ImageNet-1k", supervised_imagenet=True),
}


class Backbone(nn.Module):
    """images (B, 3, S, S) -> g (B, Dg), tokens (B, N, D)."""

    def __init__(self, name: str, img_size: int = 224, pretrained: bool = True):
        super().__init__()
        if name not in BACKBONES:
            raise KeyError(f"unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
        self.name, self.spec, self.img_size = name, BACKBONES[name], img_size
        self.grid = img_size // self.spec.patch
        if self.spec.lib == "timm":
            import timm
            kw = {"img_size": img_size} if "dinov2" in self.spec.model_id else {}
            self.model = timm.create_model(self.spec.model_id, pretrained=pretrained,
                                           num_classes=0, **kw)
            self.prefix = int(getattr(self.model, "num_prefix_tokens", 1))
            self.dim_token = int(self.model.num_features)
            self.dim_global = 2 * self.dim_token
        else:
            import torchvision
            ctor = getattr(torchvision.models, self.spec.model_id)
            m = ctor(weights=self.spec.weights if pretrained else None)
            if self.spec.model_id.startswith("resnet"):
                self.body = nn.Sequential(*list(m.children())[:-2])
                self.post = nn.Identity()
                self.dim_token = m.fc.in_features
            elif self.spec.model_id.startswith("convnext"):
                self.body = m.features
                self.post = m.classifier[0]          # LayerNorm2d of the original classifier
                self.dim_token = m.classifier[2].in_features
            else:
                raise ValueError(self.spec.model_id)
            self.dim_global = self.dim_token
        self.n_tokens = self.grid * self.grid

    def forward(self, x: torch.Tensor):
        if self.spec.lib == "timm":
            f = self.model.forward_features(x)
            cls, patches = f[:, 0], f[:, self.prefix:]
            return torch.cat([cls, patches.mean(dim=1)], dim=1), patches
        fm = self.post(self.body(x))
        tokens = fm.flatten(2).transpose(1, 2)
        return tokens.mean(dim=1), tokens

    def set_trainable(self, last_blocks: int = 0) -> int:
        """Freeze everything, then unfreeze the last ``last_blocks`` blocks (+ final norm).

        Returns the number of trainable parameters.
        """
        for p in self.parameters():
            p.requires_grad_(False)
        if last_blocks > 0:
            if self.spec.lib == "timm":
                blocks = list(self.model.blocks)[-last_blocks:]
                mods = blocks + [self.model.norm]
            elif self.spec.model_id.startswith("resnet"):
                mods = list(self.body.children())[-last_blocks:]
            else:
                mods = list(self.body.children())[-last_blocks:] + [self.post]
            for m in mods:
                for p in m.parameters():
                    p.requires_grad_(True)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> dict:
        return {"name": self.name, "model_id": self.spec.model_id, "img_size": self.img_size,
                "grid": self.grid, "n_tokens": self.n_tokens, "dim_token": self.dim_token,
                "dim_global": self.dim_global, "pretraining": self.spec.pretraining,
                "supervised_imagenet": self.spec.supervised_imagenet,
                "params": sum(p.numel() for p in self.parameters())}
