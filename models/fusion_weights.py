# -*- coding: utf-8 -*-
"""融合权重生成与 FiLM（支持 GAP 前空间调制）。"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionWeightGeneratorFromFeat(nn.Module):
    def __init__(self, feat_dim: int, num_weights: int = 3, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_weights),
        )

    def forward(self, f_img: torch.Tensor, f_clin: torch.Tensor) -> torch.Tensor:
        x = torch.cat([f_img, f_clin], dim=1)
        return F.softmax(self.mlp(x), dim=1)


class FusionWeightGenerator2(nn.Module):
    def __init__(self, f_miss_dim: int, num_weights: int = 2, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.num_weights = num_weights
        self.mlp = nn.Sequential(
            nn.Linear(f_miss_dim + 1, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_weights),
        )

    def forward(self, f_miss: torch.Tensor, r_global: torch.Tensor) -> torch.Tensor:
        r_g = r_global.unsqueeze(1)
        x = torch.cat([f_miss, r_g], dim=1)
        return F.softmax(self.mlp(x), dim=1)


class FusionWeightGenerator(nn.Module):
    def __init__(self, f_miss_dim: int, num_weights: int = 3, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(f_miss_dim + 1, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_weights),
        )

    def forward(self, f_miss: torch.Tensor, r_global: torch.Tensor) -> torch.Tensor:
        r_g = r_global.unsqueeze(1)
        x = torch.cat([f_miss, r_g], dim=1)
        return F.softmax(self.mlp(x), dim=1)


class FiLMProjection(nn.Module):
    def __init__(self, clin_dim: int, img_dim: int, use_spatial: bool = True):
        super().__init__()
        self.use_spatial = use_spatial
        self.h = nn.Sequential(
            nn.Linear(clin_dim, img_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(img_dim * 2, img_dim * 2),
        )

    def forward(
        self, f_clin: torch.Tensor, spatial_feat_shape: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.h(f_clin)
        gamma, beta = torch.chunk(out, 2, dim=1)
        gamma = torch.tanh(gamma)
        beta = torch.tanh(beta)
        if self.use_spatial and spatial_feat_shape is not None:
            B, C, D, H, W = spatial_feat_shape
            gamma = gamma.view(B, C, 1, 1, 1).expand(B, C, D, H, W)
            beta = beta.view(B, C, 1, 1, 1).expand(B, C, D, H, W)
        return gamma, beta
