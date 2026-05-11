"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
ConvNeXt backbone adapter using DINOv3 pretrained weights from timm.
"""
from typing import List, Optional, Sequence, Union

import timm
import torch
import torch.nn as nn

from ..core import register
from .hybrid_encoder import ConvNormLayer_fuse

__all__ = ["ConvNeXtAdapter"]


@register()
class ConvNeXtAdapter(nn.Module):
    """Hierarchical ConvNeXt backbone wrapper for ECDet.

    Emits feature maps at strides 8/16/32 (timm 0-indexed stages 1/2/3)
    suitable for direct consumption by HybridEncoder.
    """

    def __init__(
        self,
        name: str = "convnext_tiny.dinov3_lvd1689m",
        pretrained: bool = True,
        out_indices: Sequence[int] = (1, 2, 3),
        proj_dim: Optional[Union[int, List[int]]] = None,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.name = name
        self.out_indices = tuple(out_indices)

        self.backbone = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            out_indices=self.out_indices,
            drop_path_rate=drop_path_rate,
        )

        feat_channels = list(self.backbone.feature_info.channels())
        feat_strides = list(self.backbone.feature_info.reduction())
        assert feat_strides == [8, 16, 32], (
            f"Expected stage strides [8, 16, 32] for ECDet, got {feat_strides}. "
            f"Check out_indices={self.out_indices} for model {name!r}."
        )
        self.feat_channels = feat_channels
        self.feat_strides = feat_strides

        if proj_dim is None:
            self.projector = None
            self.out_channels = feat_channels
        else:
            dims = (
                list(proj_dim)
                if isinstance(proj_dim, (list, tuple))
                else [int(proj_dim)] * len(self.out_indices)
            )
            assert len(dims) == len(self.out_indices), (
                f"proj_dim length {len(dims)} must match out_indices length {len(self.out_indices)}"
            )
            self.projector = nn.ModuleList(
                [
                    ConvNormLayer_fuse(c_in, c_out, kernel_size=1, stride=1)
                    for c_in, c_out in zip(feat_channels, dims)
                ]
            )
            self.out_channels = dims

        self._log_loaded(pretrained)

    def _log_loaded(self, pretrained: bool) -> None:
        rank_ok = True
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank_ok = torch.distributed.get_rank() == 0
        if not rank_ok:
            return
        total = sum(p.numel() for p in self.backbone.parameters())
        checksum = sum(p.detach().float().abs().sum().item() for p in self.backbone.parameters())
        status = "pretrained" if pretrained else "RANDOM INIT"
        print(
            "=" * 80 + "\n"
            f"ConvNeXtAdapter loaded: {self.name} ({status})\n"
            f"  out_indices: {self.out_indices}\n"
            f"  feat_channels: {self.feat_channels}\n"
            f"  feat_strides:  {self.feat_strides}\n"
            f"  out_channels:  {self.out_channels}\n"
            f"  backbone params: {total:,}\n"
            f"  param-abs-sum checksum: {checksum:.4f}\n"
            + "=" * 80
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.backbone(x)
        if self.projector is None:
            return list(feats)
        return [proj(f) for proj, f in zip(self.projector, feats)]
