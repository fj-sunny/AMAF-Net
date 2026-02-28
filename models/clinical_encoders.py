# -*- coding: utf-8 -*-
"""临床编码与缺失模式 MLP。"""
from typing import Tuple

import torch
import torch.nn as nn


class MissingPatternMLP(nn.Module):
    def __init__(self, d: int, out_dim: int = 64, hidden: Tuple[int, ...] = (128, 64), dropout: float = 0.2):
        super().__init__()
        in_dim = d + 1
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.to_out = nn.Linear(prev, out_dim)

    def forward(self, R: torch.Tensor, r_global: torch.Tensor) -> torch.Tensor:
        r_g = r_global.unsqueeze(1)
        x = torch.cat([R, r_g], dim=1)
        return self.to_out(self.mlp(x))


class ClinicalEncoderMissingAware(nn.Module):
    def __init__(self, d: int, out_dim: int = 256, hidden: Tuple[int, ...] = (256, 128), dropout: float = 0.2):
        super().__init__()
        in_dim = 2 * d
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.to_emb = nn.Linear(prev, out_dim)

    def forward(self, C_tilde: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        x = torch.cat([C_tilde, R], dim=1)
        return self.to_emb(self.encoder(x))


class ClinicalEncoderCtildeOnly(nn.Module):
    def __init__(self, d: int, out_dim: int = 256, hidden: Tuple[int, ...] = (256, 128), dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.to_emb = nn.Linear(prev, out_dim)

    def forward(self, C_tilde: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        return self.to_emb(self.encoder(C_tilde))


class MissingPatternMLP_RGlobalOnly(nn.Module):
    def __init__(self, out_dim: int = 64, hidden: Tuple[int, ...] = (64, 32), dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = 1
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.to_out = nn.Linear(prev, out_dim)

    def forward(self, R: torch.Tensor, r_global: torch.Tensor) -> torch.Tensor:
        r_g = r_global.unsqueeze(1)
        return self.to_out(self.mlp(r_g))
