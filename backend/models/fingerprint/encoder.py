"""
Shared lightweight per-representation encoder — proposal §4.1, §6.

Deliberately small (MobileNetV3-Small scale, ~2-3M params), not another
EfficientNet-B4: 6 of these run per image (one per representation) plus
patch-level reuse (§4.3), and this project's training hardware is a 4.3GB
laptop GPU (see PROJECT_SPEC.md training notes) — a heavy backbone per
representation would blow that budget for a marginal gain the proposal's
own ablation (§10) might show isn't needed anyway.
"""

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int, stride: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
    )


class SmallEncoder(nn.Module):
    """
    4 strided conv blocks + global average pool -> embedding vector.
    Input: any HxW >= 32 (gets downsampled 16x internally); any channel count.
    """

    def __init__(self, in_channels: int, embed_dim: int = 128, widths=(32, 64, 128, 128)):
        super().__init__()
        chs = [in_channels] + list(widths)
        self.blocks = nn.Sequential(*[
            _conv_block(chs[i], chs[i + 1]) for i in range(len(widths))
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(widths[-1], embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] -> [B, embed_dim]"""
        feat = self.blocks(x)
        pooled = self.pool(feat).flatten(1)
        return self.proj(pooled)
