"""
影像与临床 Backbones 定义

说明：
- 为保持与现有训练/评估脚本兼容，这里的大部分 3D ResNet / 临床编码器直接从
  `train_missing_aware_fusion.py` 导入并做轻量封装。
- 额外实现论文表中的若干基线模型：
  - Image-only: 3D ResNet18, 3D U-Net, Image ViT
  - Clinical-only: MLP, MLP+Missingness, FT-Transformer 风格编码器

所有 backbone 在多模态场景下都遵循统一接口：
    forward(...) -> 特征向量 (B, D)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# 使用本包内部分解后的 3D ResNet 与临床编码器
# -------------------------------------------------------------------------

from .resnet3d import ResNet18_3D_Encoder as _ResNet18_3D_Encoder
from .clinical_encoders import (
    ClinicalEncoderMissingAware as _ClinicalEncoderMissingAware,
    ClinicalEncoderCtildeOnly as _ClinicalEncoderCtildeOnly,
    MissingPatternMLP as _MissingPatternMLP,
)


# -------------------------------------------------------------------------
# Image-only backbones
# -------------------------------------------------------------------------

class ImageResNet18(nn.Module):
    """
    Image-only 3D ResNet18 baseline.

    - 输入: x_ct ∈ R^{B×1×D×H×W}
    - 输出: logits ∈ R^{B×C}
    - 额外返回 aux["f_img"] 以便与多模态评估逻辑兼容
    """

    def __init__(self, num_classes: int, feat_dim: int = 512):
        super().__init__()
        self.encoder = _ResNet18_3D_Encoder(in_channels=1)
        self.feat_dim = feat_dim
        # 原 encoder 输出 512 维，可选再线性变换到 feat_dim
        self.proj = nn.Identity() if feat_dim == 512 else nn.Linear(512, feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x_ct: torch.Tensor, *args, **kwargs):
        feat = self.encoder(x_ct)  # (B,512)
        feat = self.proj(feat)
        logits = self.classifier(feat)
        B = x_ct.size(0)
        device = x_ct.device
        aux = {
            "f_img": feat,
            "f_clin": torch.zeros_like(feat),
            "f_int": torch.zeros_like(feat),
            "alpha_img": torch.ones((B, 1), device=device, dtype=feat.dtype),
            "alpha_clin": torch.zeros((B, 1), device=device, dtype=feat.dtype),
            "alpha_int": torch.zeros((B, 1), device=device, dtype=feat.dtype),
        }
        return logits, aux


class DoubleConv3D(nn.Module):
    """3D U-Net 基本卷积块: (Conv3D → BN → ReLU) × 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet3DEncoder(nn.Module):
    """
    轻量级 3D U-Net encoder，用于 Image-only baseline。
    仅保留编码部分 + 最后一层特征 GAP 得到体素级特征。
    """

    def __init__(self, in_channels: int = 1, base_ch: int = 32):
        super().__init__()
        self.down1 = DoubleConv3D(in_channels, base_ch)
        self.pool1 = nn.MaxPool3d(2)
        self.down2 = DoubleConv3D(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.down3 = DoubleConv3D(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.down4 = DoubleConv3D(base_ch * 4, base_ch * 8)

    @property
    def out_channels(self) -> int:
        return 8 * 32

    def forward(self, x: torch.Tensor, return_spatial: bool = False) -> torch.Tensor:
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.down4(self.pool3(x3))  # (B, C, D',H',W')
        if return_spatial:
            return x4
        return F.adaptive_avg_pool3d(x4, 1).flatten(1)


class ImageUNet3D(nn.Module):
    """
    Image-only 3D U-Net baseline.
    接 UNet3DEncoder 后接 GAP+FC 分类。
    """

    def __init__(self, num_classes: int, feat_dim: int = 256, base_ch: int = 32):
        super().__init__()
        self.encoder = UNet3DEncoder(in_channels=1, base_ch=base_ch)
        self.proj = nn.Linear(self.encoder.out_channels, feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x_ct: torch.Tensor, *args, **kwargs):
        feat_spatial = self.encoder(x_ct, return_spatial=True)
        feat = F.adaptive_avg_pool3d(feat_spatial, 1).flatten(1)
        feat = self.proj(feat)
        logits = self.classifier(feat)
        B = x_ct.size(0)
        device = x_ct.device
        aux = {
            "f_img": feat,
            "f_clin": torch.zeros_like(feat),
            "f_int": torch.zeros_like(feat),
            "alpha_img": torch.ones((B, 1), device=device, dtype=feat.dtype),
            "alpha_clin": torch.zeros((B, 1), device=device, dtype=feat.dtype),
            "alpha_int": torch.zeros((B, 1), device=device, dtype=feat.dtype),
        }
        return logits, aux


class PatchEmbed3D(nn.Module):
    """简单 3D patch embedding，用于 ViT baseline。"""

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 256,
        patch_size: Tuple[int, int, int] = (16, 16, 16),
    ):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,D,H,W) -> (B,embed_dim,N_patches)
        x = self.proj(x)  # (B,E,D',H',W')
        x = x.flatten(2).transpose(1, 2)  # (B,N,E)
        return x


class SimpleViT3D(nn.Module):
    """
    非常简化的 3D ViT baseline（Image-only）。
    仅用于作为 “Image-only ViT” 对比实验，不追求 SOTA 实现。
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed3D(1, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, embed_dim))  # 上限，实际会裁剪
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x_ct: torch.Tensor, *args, **kwargs):
        B = x_ct.size(0)
        x = self.patch_embed(x_ct)  # (B,N,E)
        n = x.size(1)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B,N+1,E)
        # 简单位置编码（截断/重复）
        if self.pos_embed.size(1) >= x.size(1):
            pos = self.pos_embed[:, : x.size(1), :]
        else:
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=x.size(1),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        x = x + pos
        x = self.encoder(x)
        cls_feat = self.norm(x[:, 0])
        logits = self.head(cls_feat)
        device = x_ct.device
        aux = {
            "f_img": cls_feat,
            "f_clin": torch.zeros_like(cls_feat),
            "f_int": torch.zeros_like(cls_feat),
            "alpha_img": torch.ones((B, 1), device=device, dtype=cls_feat.dtype),
            "alpha_clin": torch.zeros((B, 1), device=device, dtype=cls_feat.dtype),
            "alpha_int": torch.zeros((B, 1), device=device, dtype=cls_feat.dtype),
        }
        return logits, aux


# -------------------------------------------------------------------------
# Clinical-only backbones
# -------------------------------------------------------------------------

class ClinicalMLP(nn.Module):
    """Clinical-only MLP baseline: 仅使用 C_tilde 作为输入。"""

    def __init__(self, clin_dim: int, num_classes: int, hidden: Tuple[int, ...] = (256, 128), dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = clin_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x_ct: torch.Tensor, C_tilde: torch.Tensor, R: torch.Tensor, r_global: torch.Tensor):
        feat = self.mlp(C_tilde)
        logits = self.head(feat)
        B = x_ct.size(0)
        device = x_ct.device
        aux = {
            "f_img": torch.zeros_like(feat),
            "f_clin": feat,
            "f_int": torch.zeros_like(feat),
            "alpha_img": torch.zeros((B, 1), device=device, dtype=feat.dtype),
            "alpha_clin": torch.ones((B, 1), device=device, dtype=feat.dtype),
            "alpha_int": torch.zeros((B, 1), device=device, dtype=feat.dtype),
        }
        return logits, aux


class ClinicalMLPWithMissing(nn.Module):
    """
    Clinical-only MLP+Missingness baseline:
    输入拼接 [C_tilde; R; r_global]，显式利用缺失模式。
    """

    def __init__(self, clin_dim: int, num_classes: int, hidden: Tuple[int, ...] = (256, 128), dropout: float = 0.2):
        super().__init__()
        in_dim = 2 * clin_dim + 1
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x_ct: torch.Tensor, C_tilde: torch.Tensor, R: torch.Tensor, r_global: torch.Tensor):
        r_g = r_global.unsqueeze(1)
        x = torch.cat([C_tilde, R, r_g], dim=1)
        feat = self.mlp(x)
        logits = self.head(feat)
        B = x_ct.size(0)
        device = x_ct.device
        aux = {
            "f_img": torch.zeros_like(feat),
            "f_clin": feat,
            "f_int": torch.zeros_like(feat),
            "alpha_img": torch.zeros((B, 1), device=device, dtype=feat.dtype),
            "alpha_clin": torch.ones((B, 1), device=device, dtype=feat.dtype),
            "alpha_int": torch.zeros((B, 1), device=device, dtype=feat.dtype),
        }
        return logits, aux


class FTTransformerClinical(nn.Module):
    """
    简化版 FT-Transformer 风格临床编码器：
    - 将每个特征视为一个 token，拼接数值 embedding 与可学习列 embedding。
    - 多层 TransformerEncoder 后做 tokens 的 mean pooling 得到全局表征。
    仅作为 “Clinical-only FT-Transformer” 对比，不追求完整论文细节。
    """

    def __init__(
        self,
        clin_dim: int,
        num_classes: int,
        d_token: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.clin_dim = clin_dim
        self.value_proj = nn.Linear(1, d_token)
        self.col_embed = nn.Embedding(clin_dim, d_token)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, num_classes)

    def forward(self, x_ct: torch.Tensor, C_tilde: torch.Tensor, R: torch.Tensor, r_global: torch.Tensor):
        B, D = C_tilde.shape
        assert D == self.clin_dim
        v = C_tilde.view(B, D, 1)
        v_emb = self.value_proj(v)
        idx = torch.arange(D, device=C_tilde.device)
        col_e = self.col_embed(idx)[None, :, :].expand(B, -1, -1)
        tokens = v_emb + col_e
        tokens = self.encoder(tokens)
        feat = self.norm(tokens.mean(dim=1))
        logits = self.head(feat)
        B = x_ct.size(0)
        device = x_ct.device
        aux = {
            "f_img": torch.zeros_like(feat),
            "f_clin": feat,
            "f_int": torch.zeros_like(feat),
            "alpha_img": torch.zeros((B, 1), device=device, dtype=feat.dtype),
            "alpha_clin": torch.ones((B, 1), device=device, dtype=feat.dtype),
            "alpha_int": torch.zeros((B, 1), device=device, dtype=feat.dtype),
        }
        return logits, aux


# -------------------------------------------------------------------------
# 便捷别名，供 registry 使用
# -------------------------------------------------------------------------

ResNet18_3D_Encoder = _ResNet18_3D_Encoder
ClinicalEncoderMissingAware = _ClinicalEncoderMissingAware
ClinicalEncoderCtildeOnly = _ClinicalEncoderCtildeOnly
MissingPatternMLP = _MissingPatternMLP

