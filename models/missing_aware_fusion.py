# -*- coding: utf-8 -*-
"""缺失感知三路融合主模型与消融变体（FiLM 在 ResNet18 layer4 后、GAP 前）。"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet3d import ResNet18_3D_Encoder
from .clinical_encoders import (
    ClinicalEncoderMissingAware,
    ClinicalEncoderCtildeOnly,
    MissingPatternMLP,
    MissingPatternMLP_RGlobalOnly,
)
from .fusion_weights import (
    FiLMProjection,
    FusionWeightGenerator,
    FusionWeightGenerator2,
    FusionWeightGeneratorFromFeat,
)


class MissingAwareFusionModel(nn.Module):
    """三路融合：f_img（纯图像）+ f_clin（临床）+ f_int（FiLM 在 layer4 后 GAP 前调制）。"""

    def __init__(
        self,
        clin_dim: int,
        num_classes: int = 4,
        df: int = 256,
        clin_emb_dim: int = 256,
        f_miss_dim: int = 64,
    ):
        super().__init__()
        self.df = df
        self.clin_emb_dim = clin_emb_dim
        self.ct_encoder = ResNet18_3D_Encoder(in_channels=1)
        self.ct_proj = nn.Linear(512, df)
        self.missing_mlp = MissingPatternMLP(d=clin_dim, out_dim=f_miss_dim, hidden=(128, 64), dropout=0.2)
        self.clin_encoder = ClinicalEncoderMissingAware(d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2)
        self.fusion_weight_gen = FusionWeightGenerator(f_miss_dim=f_miss_dim, num_weights=3, hidden=64, dropout=0.1)
        self.film = FiLMProjection(clin_dim=clin_emb_dim, img_dim=512, use_spatial=True)
        self.classifier = nn.Linear(df, num_classes)

    def forward(
        self,
        x_ct: torch.Tensor,
        C_tilde: torch.Tensor,
        R: torch.Tensor,
        r_global: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        X_spatial = self.ct_encoder(x_ct, return_spatial=True)
        f_clin = self.clin_encoder(C_tilde, R)
        gamma, beta = self.film(f_clin, spatial_feat_shape=X_spatial.shape)
        X_int_spatial = gamma * X_spatial + beta
        f_img = F.adaptive_avg_pool3d(X_spatial, 1).flatten(1)
        f_img = self.ct_proj(f_img)
        f_int = F.adaptive_avg_pool3d(X_int_spatial, 1).flatten(1)
        f_int = self.ct_proj(f_int)
        f_miss = self.missing_mlp(R, r_global)
        alphas = self.fusion_weight_gen(f_miss, r_global)
        alpha_img, alpha_clin, alpha_int = alphas[:, 0:1], alphas[:, 1:2], alphas[:, 2:3]
        f = alpha_img * f_img + alpha_clin * f_clin + alpha_int * f_int
        logits = self.classifier(f)
        aux = {
            "f_img": f_img,
            "f_clin": f_clin,
            "f_int": f_int,
            "alpha_img": alpha_img,
            "alpha_clin": alpha_clin,
            "alpha_int": alpha_int,
        }
        return logits, aux


ABLATION_NAMES = ["A1", "A2", "A3", "B1", "B2", "C1", "C2", "E1"]
ABLATION_DESCRIPTIONS = {
    "A1": "w/o Missing Mask (不显式建模 R/r_global/missing MLP)",
    "A2": "w/o Missing Pattern Encoder (只用 r_global 控制融合)",
    "A3": "Fixed Fusion Weights (1/3, 1/3, 1/3)",
    "B1": "w/o Interaction Term (无 FiLM, 无 f_int)",
    "B2": "Interaction w/o Gating (alpha_int 固定 1/3)",
    "C1": "w/o Direct Clinical Branch (无 alpha_clin*f_clin)",
    "C2": "Simple Concat (concat+MLP, 无 FiLM/动态 alpha)",
    "D1": "w/o Class-balanced Loss (普通 CE, 无 class weight/sampler)",
    "D2": "w/o Missing-aware Training (训练时不随机 mask)",
    "E1": "w/o Pure Image Branch (无 alpha_img*f_img, 仅临床+调制图像融合)",
}


class AblationFusionModel(nn.Module):
    """消融变体：A1–A3, B1–B2, C1–C2, E1。forward 与 MissingAwareFusionModel 一致。"""

    def __init__(
        self,
        ablation: str,
        clin_dim: int,
        num_classes: int = 4,
        df: int = 256,
        clin_emb_dim: int = 256,
        f_miss_dim: int = 64,
    ):
        super().__init__()
        self.ablation = ablation
        self.df = df
        self.clin_emb_dim = clin_emb_dim
        self.ct_encoder = ResNet18_3D_Encoder(in_channels=1)
        self.ct_proj = nn.Linear(512, df)
        # 消融分支使用向量级 FiLM (img_dim=df)，非空间
        if ablation == "A1":
            self.clin_encoder = ClinicalEncoderCtildeOnly(d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2)
            self.missing_mlp = None
            self.fusion_weight_gen = FusionWeightGeneratorFromFeat(feat_dim=df + clin_emb_dim, num_weights=3, hidden=64, dropout=0.1)
            self.film = FiLMProjection(clin_dim=clin_emb_dim, img_dim=df, use_spatial=False)
            self.use_f_int, self.use_alpha_clin = True, True
            self.fixed_alphas, self.alpha_int_fixed = None, None
        elif ablation == "A2":
            self.clin_encoder = ClinicalEncoderMissingAware(d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2)
            self.missing_mlp = MissingPatternMLP_RGlobalOnly(out_dim=f_miss_dim, hidden=(64, 32), dropout=0.2)
            self.fusion_weight_gen = FusionWeightGenerator(f_miss_dim=f_miss_dim, num_weights=3, hidden=64, dropout=0.1)
            self.film = FiLMProjection(clin_dim=clin_emb_dim, img_dim=df, use_spatial=False)
            self.use_f_int, self.use_alpha_clin = True, True
            self.fixed_alphas, self.alpha_int_fixed = None, None
        elif ablation == "A3":
            self.clin_encoder = ClinicalEncoderMissingAware(d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2)
            self.missing_mlp = MissingPatternMLP(d=clin_dim, out_dim=f_miss_dim, hidden=(128, 64), dropout=0.2)
            self.fusion_weight_gen = None
            self.film = FiLMProjection(clin_dim=clin_emb_dim, img_dim=df, use_spatial=False)
            self.use_f_int, self.use_alpha_clin = True, True
            self.fixed_alphas, self.alpha_int_fixed = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), None
        elif ablation == "B1":
            self.clin_encoder = ClinicalEncoderMissingAware(d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2)
            self.missing_mlp = MissingPatternMLP(d=clin_dim, out_dim=f_miss_dim, hidden=(128, 64), dropout=0.2)
            self.fusion_weight_gen = FusionWeightGenerator2(f_miss_dim=f_miss_dim, num_weights=2, hidden=64, dropout=0.1)
            self.film = None
            self.use_f_int, self.use_alpha_clin = False, True
            self.fixed_alphas, self.alpha_int_fixed = None, None
        elif ablation == "B2":
            self.clin_encoder = ClinicalEncoderMissingAware(d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2)
            self.missing_mlp = MissingPatternMLP(d=clin_dim, out_dim=f_miss_dim, hidden=(128, 64), dropout=0.2)
            self.fusion_weight_gen = FusionWeightGenerator2(f_miss_dim=f_miss_dim, num_weights=2, hidden=64, dropout=0.1)
            self.film = FiLMProjection(clin_dim=clin_emb_dim, img_dim=df, use_spatial=False)
            self.use_f_int, self.use_alpha_clin = True, True
            self.fixed_alphas, self.alpha_int_fixed = None, 1.0 / 3.0
        elif ablation == "C1":
            self.clin_encoder = ClinicalEncoderMissingAware(d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2)
            self.missing_mlp = MissingPatternMLP(d=clin_dim, out_dim=f_miss_dim, hidden=(128, 64), dropout=0.2)
            self.fusion_weight_gen = FusionWeightGenerator2(f_miss_dim=f_miss_dim, num_weights=2, hidden=64, dropout=0.1)
            self.film = FiLMProjection(clin_dim=clin_emb_dim, img_dim=df, use_spatial=False)
            self.use_f_int, self.use_alpha_clin = True, False
            self.fixed_alphas, self.alpha_int_fixed = None, None
        elif ablation == "C2":
            self.clin_encoder = ClinicalEncoderMissingAware(d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2)
            self.missing_mlp = None
            self.fusion_weight_gen = None
            self.film = None
            self.use_f_int = self.use_alpha_clin = False
            self.fixed_alphas = self.alpha_int_fixed = None
            self.fusion_concat = nn.Sequential(
                nn.Linear(df + clin_emb_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )
            self.classifier = None
        elif ablation == "E1":
            self.clin_encoder = ClinicalEncoderMissingAware(d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2)
            self.missing_mlp = MissingPatternMLP(d=clin_dim, out_dim=f_miss_dim, hidden=(128, 64), dropout=0.2)
            self.fusion_weight_gen = FusionWeightGenerator2(f_miss_dim=f_miss_dim, num_weights=2, hidden=64, dropout=0.1)
            self.film = FiLMProjection(clin_dim=clin_emb_dim, img_dim=df, use_spatial=False)
            self.use_f_int, self.use_alpha_clin = True, True
            self.fixed_alphas, self.alpha_int_fixed = None, None
            self.use_alpha_img = False
        else:
            raise ValueError(f"Unknown ablation: {ablation}")
        if ablation != "C2":
            self.classifier = nn.Linear(df, num_classes)
            self.fusion_concat = None
        if not hasattr(self, "use_alpha_img"):
            self.use_alpha_img = True

    def forward(
        self,
        x_ct: torch.Tensor,
        C_tilde: torch.Tensor,
        R: torch.Tensor,
        r_global: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        B = x_ct.size(0)
        device = x_ct.device
        f_img = self.ct_proj(self.ct_encoder(x_ct))
        f_clin = self.clin_encoder(C_tilde, R)
        if self.ablation == "C2":
            f = torch.cat([f_img, f_clin], dim=1)
            logits = self.fusion_concat(f)
            return logits, {
                "f_img": f_img, "f_clin": f_clin, "f_int": None,
                "alpha_img": torch.full((B, 1), 0.5, device=device, dtype=f_img.dtype),
                "alpha_clin": torch.full((B, 1), 0.5, device=device, dtype=f_img.dtype),
                "alpha_int": torch.zeros((B, 1), device=device, dtype=f_img.dtype),
            }
        if self.fixed_alphas is not None:
            a1, a2, a3 = self.fixed_alphas
            alpha_img = torch.full((B, 1), a1, device=device, dtype=f_img.dtype)
            alpha_clin = torch.full((B, 1), a2, device=device, dtype=f_img.dtype)
            alpha_int = torch.full((B, 1), a3, device=device, dtype=f_img.dtype)
        elif self.ablation == "A1":
            alphas = self.fusion_weight_gen(f_img, f_clin)
            alpha_img, alpha_clin, alpha_int = alphas[:, 0:1], alphas[:, 1:2], alphas[:, 2:3]
        elif self.alpha_int_fixed is not None:
            f_miss = self.missing_mlp(R, r_global)
            ab = self.fusion_weight_gen(f_miss, r_global)
            a_img, a_clin = ab[:, 0:1], ab[:, 1:2]
            remainder = 1.0 - self.alpha_int_fixed
            s = (a_img + a_clin).clamp(min=1e-8)
            alpha_img = remainder * a_img / s
            alpha_clin = remainder * a_clin / s
            alpha_int = torch.full((B, 1), self.alpha_int_fixed, device=device, dtype=f_img.dtype)
        else:
            f_miss = self.missing_mlp(R, r_global)
            alphas = self.fusion_weight_gen(f_miss, r_global)
            if alphas.size(1) == 2:
                a_img, a_clin = alphas[:, 0:1], alphas[:, 1:2]
                if self.use_f_int and self.use_alpha_clin:
                    if hasattr(self, "use_alpha_img") and not self.use_alpha_img:
                        s = (a_clin + a_img).clamp(min=1e-8)
                        alpha_clin = a_clin / s
                        alpha_int = a_img / s
                        alpha_img = torch.zeros((B, 1), device=device, dtype=f_img.dtype)
                    else:
                        alpha_img, alpha_clin = a_img, a_clin
                        alpha_int = torch.zeros((B, 1), device=device, dtype=f_img.dtype)
                elif self.use_f_int and not self.use_alpha_clin:
                    alpha_img, alpha_int = a_img, a_clin
                    alpha_clin = torch.zeros((B, 1), device=device, dtype=f_img.dtype)
                else:
                    alpha_img, alpha_clin = a_img, a_clin
                    alpha_int = torch.zeros((B, 1), device=device, dtype=f_img.dtype)
            else:
                alpha_img, alpha_clin, alpha_int = alphas[:, 0:1], alphas[:, 1:2], alphas[:, 2:3]
        if self.use_f_int and self.film is not None:
            gamma, beta = self.film(f_clin)
            f_int = gamma * f_img + beta
        else:
            f_int = torch.zeros_like(f_img)
        if hasattr(self, "use_alpha_img") and not self.use_alpha_img:
            f = alpha_clin * f_clin + alpha_int * f_int
            alpha_img = torch.zeros((B, 1), device=device, dtype=f_img.dtype)
        elif self.use_alpha_clin:
            f = alpha_img * f_img + alpha_clin * f_clin + alpha_int * f_int
        else:
            f = alpha_img * f_img + alpha_int * f_int
            alpha_clin = torch.zeros((B, 1), device=device, dtype=f_img.dtype)
        logits = self.classifier(f)
        return logits, {
            "f_img": f_img, "f_clin": f_clin, "f_int": f_int,
            "alpha_img": alpha_img, "alpha_clin": alpha_clin, "alpha_int": alpha_int,
        }


def build_ablation_model(
    ablation: Optional[str],
    clin_dim: int,
    num_classes: int = 4,
    df: int = 256,
    clin_emb_dim: int = 256,
    f_miss_dim: int = 64,
) -> nn.Module:
    if ablation is None or ablation == "full":
        return MissingAwareFusionModel(
            clin_dim=clin_dim, num_classes=num_classes, df=df,
            clin_emb_dim=clin_emb_dim, f_miss_dim=f_miss_dim,
        )
    return AblationFusionModel(
        ablation=ablation, clin_dim=clin_dim, num_classes=num_classes,
        df=df, clin_emb_dim=clin_emb_dim, f_miss_dim=f_miss_dim,
    )
