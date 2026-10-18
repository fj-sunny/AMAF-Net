# -*- coding: utf-8 -*-
"""带融合特征输出的 ProCo 模型。

主模型与 A1–E1 消融继承本包的 MissingAwareFusion / AblationFusion，
只在 aux 中补充 f_fusion。E1 系列来自 train_missing_aware_fusion_proco.py。
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .clinical_encoders import (
    ClinicalEncoderCtildeOnly,
    ClinicalEncoderMissingAware,
    MissingPatternMLP,
    MissingPatternMLP_RGlobalOnly,
)
from .fusion_weights import (
    FiLMProjection,
    FusionWeightGenerator,
    FusionWeightGenerator2,
    FusionWeightGeneratorFromFeat,
)
from .missing_aware_fusion import AblationFusionModel, MissingAwareFusionModel
from .resnet3d import ResNet18_3D_Encoder


class MissingAwareFusionModelProCo(MissingAwareFusionModel):
    """在主模型 aux 中显式返回融合特征，供 ProCo 使用。"""

    def forward(self, x_ct, C_tilde, R, r_global) -> Tuple[torch.Tensor, dict]:
        logits, aux = super().forward(x_ct, C_tilde, R, r_global)
        aux["f_fusion"] = (
            aux["alpha_img"] * aux["f_img"]
            + aux["alpha_clin"] * aux["f_clin"]
            + aux["alpha_int"] * aux["f_int"]
        )
        return logits, aux


class AblationFusionModelProCo(AblationFusionModel):
    """消融模型同样返回 f_fusion。C2 的融合特征是拼接向量。"""

    def forward(self, x_ct, C_tilde, R, r_global) -> Tuple[torch.Tensor, dict]:
        b = x_ct.size(0)
        device = x_ct.device
        f_img = self.ct_proj(self.ct_encoder(x_ct))
        f_clin = self.clin_encoder(C_tilde, R)

        if self.ablation == "C2":
            fused = torch.cat([f_img, f_clin], dim=1)
            logits = self.fusion_concat(fused)
            aux = {
                "f_img": f_img,
                "f_clin": f_clin,
                "f_int": torch.zeros_like(f_img),
                "f_fusion": fused,
                "alpha_img": torch.full((b, 1), 0.5, device=device, dtype=f_img.dtype),
                "alpha_clin": torch.full((b, 1), 0.5, device=device, dtype=f_img.dtype),
                "alpha_int": torch.zeros((b, 1), device=device, dtype=f_img.dtype),
            }
            return logits, aux

        if self.fixed_alphas is not None:
            a1, a2, a3 = self.fixed_alphas
            alpha_img = torch.full((b, 1), a1, device=device, dtype=f_img.dtype)
            alpha_clin = torch.full((b, 1), a2, device=device, dtype=f_img.dtype)
            alpha_int = torch.full((b, 1), a3, device=device, dtype=f_img.dtype)
        elif self.ablation == "A1":
            alphas = self.fusion_weight_gen(f_img, f_clin)
            alpha_img, alpha_clin, alpha_int = alphas[:, 0:1], alphas[:, 1:2], alphas[:, 2:3]
        elif self.alpha_int_fixed is not None:
            f_miss = self.missing_mlp(R, r_global)
            ab = self.fusion_weight_gen(f_miss, r_global)
            a_img, a_clin = ab[:, 0:1], ab[:, 1:2]
            remainder = 1.0 - self.alpha_int_fixed
            scale = (a_img + a_clin).clamp(min=1e-8)
            alpha_img = remainder * a_img / scale
            alpha_clin = remainder * a_clin / scale
            alpha_int = torch.full((b, 1), self.alpha_int_fixed, device=device, dtype=f_img.dtype)
        else:
            f_miss = self.missing_mlp(R, r_global)
            alphas = self.fusion_weight_gen(f_miss, r_global)
            if alphas.size(1) == 2:
                a_img, a_clin = alphas[:, 0:1], alphas[:, 1:2]
                if self.use_f_int and self.use_alpha_clin:
                    if hasattr(self, "use_alpha_img") and not self.use_alpha_img:
                        scale = (a_clin + a_img).clamp(min=1e-8)
                        alpha_clin = a_clin / scale
                        alpha_int = a_img / scale
                        alpha_img = torch.zeros((b, 1), device=device, dtype=f_img.dtype)
                    else:
                        alpha_img, alpha_clin = a_img, a_clin
                        alpha_int = torch.zeros((b, 1), device=device, dtype=f_img.dtype)
                elif self.use_f_int and not self.use_alpha_clin:
                    alpha_img, alpha_int = a_img, a_clin
                    alpha_clin = torch.zeros((b, 1), device=device, dtype=f_img.dtype)
                else:
                    alpha_img, alpha_clin = a_img, a_clin
                    alpha_int = torch.zeros((b, 1), device=device, dtype=f_img.dtype)
            else:
                alpha_img, alpha_clin, alpha_int = alphas[:, 0:1], alphas[:, 1:2], alphas[:, 2:3]

        if self.use_f_int and self.film is not None:
            gamma, beta = self.film(f_clin)
            f_int = gamma * f_img + beta
        else:
            f_int = torch.zeros_like(f_img)

        if hasattr(self, "use_alpha_img") and not self.use_alpha_img:
            f_fusion = alpha_clin * f_clin + alpha_int * f_int
            alpha_img = torch.zeros((b, 1), device=device, dtype=f_img.dtype)
        elif self.use_alpha_clin:
            f_fusion = alpha_img * f_img + alpha_clin * f_clin + alpha_int * f_int
        else:
            f_fusion = alpha_img * f_img + alpha_int * f_int
            alpha_clin = torch.zeros((b, 1), device=device, dtype=f_img.dtype)

        logits = self.classifier(f_fusion)
        aux = {
            "f_img": f_img,
            "f_clin": f_clin,
            "f_int": f_int,
            "f_fusion": f_fusion,
            "alpha_img": alpha_img,
            "alpha_clin": alpha_clin,
            "alpha_int": alpha_int,
        }
        return logits, aux


class E1MainModel(nn.Module):
    """无纯图像分支：f = α_clin·f_clin + α_int·f_int。"""

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
        self.clin_encoder = ClinicalEncoderMissingAware(
            d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2
        )
        self.missing_mlp = MissingPatternMLP(d=clin_dim, out_dim=f_miss_dim, hidden=(128, 64), dropout=0.2)
        self.fusion_weight_gen = FusionWeightGenerator2(
            f_miss_dim=f_miss_dim, num_weights=2, hidden=64, dropout=0.1
        )
        self.film = FiLMProjection(clin_dim=clin_emb_dim, img_dim=df, use_spatial=False)
        self.classifier = nn.Linear(df, num_classes)

    def forward(self, x_ct, C_tilde, R, r_global):
        b = x_ct.size(0)
        device = x_ct.device
        f_img = self.ct_proj(self.ct_encoder(x_ct))
        f_clin = self.clin_encoder(C_tilde, R)
        gamma, beta = self.film(f_clin)
        f_int = gamma * f_img + beta
        f_miss = self.missing_mlp(R, r_global)
        alphas = self.fusion_weight_gen(f_miss, r_global)
        alpha_clin, alpha_int = alphas[:, 0:1], alphas[:, 1:2]
        scale = (alpha_clin + alpha_int).clamp(min=1e-8)
        alpha_clin = alpha_clin / scale
        alpha_int = alpha_int / scale
        f_fusion = alpha_clin * f_clin + alpha_int * f_int
        logits = self.classifier(f_fusion)
        aux = {
            "f_img": f_img,
            "f_clin": f_clin,
            "f_int": f_int,
            "f_fusion": f_fusion,
            "alpha_img": torch.zeros((b, 1), device=device, dtype=f_img.dtype),
            "alpha_clin": alpha_clin,
            "alpha_int": alpha_int,
        }
        return logits, aux


class E1AblationModel(nn.Module):
    """E1 上的缺失掩码 / 权重 / 临床分支消融。"""

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
        if ablation == "E1-A1":
            self.clin_encoder = ClinicalEncoderCtildeOnly(
                d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2
            )
            self.missing_mlp = None
            self.fusion_weight_gen = FusionWeightGeneratorFromFeat(
                feat_dim=df + clin_emb_dim, num_weights=2, hidden=64, dropout=0.1
            )
            self.film = FiLMProjection(clin_dim=clin_emb_dim, img_dim=df, use_spatial=False)
        elif ablation == "E1-A2":
            self.clin_encoder = ClinicalEncoderMissingAware(
                d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2
            )
            self.missing_mlp = MissingPatternMLP_RGlobalOnly(out_dim=f_miss_dim, hidden=(64, 32), dropout=0.2)
            self.fusion_weight_gen = FusionWeightGenerator(
                f_miss_dim=f_miss_dim, num_weights=2, hidden=64, dropout=0.1
            )
            self.film = FiLMProjection(clin_dim=clin_emb_dim, img_dim=df, use_spatial=False)
        elif ablation == "E1-A3":
            self.clin_encoder = ClinicalEncoderMissingAware(
                d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2
            )
            self.missing_mlp = MissingPatternMLP(d=clin_dim, out_dim=f_miss_dim, hidden=(128, 64), dropout=0.2)
            self.fusion_weight_gen = None
            self.film = FiLMProjection(clin_dim=clin_emb_dim, img_dim=df, use_spatial=False)
        elif ablation == "E1-C1":
            self.clin_encoder = ClinicalEncoderMissingAware(
                d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2
            )
            self.missing_mlp = None
            self.fusion_weight_gen = None
            self.film = FiLMProjection(clin_dim=clin_emb_dim, img_dim=df, use_spatial=False)
        else:
            raise ValueError(f"Unknown E1 ablation: {ablation}")
        self.classifier = nn.Linear(df, num_classes)

    def forward(self, x_ct, C_tilde, R, r_global):
        b = x_ct.size(0)
        device = x_ct.device
        f_img = self.ct_proj(self.ct_encoder(x_ct))
        f_clin = self.clin_encoder(C_tilde, R)
        gamma, beta = self.film(f_clin)
        f_int = gamma * f_img + beta
        if self.ablation == "E1-A1":
            alphas = self.fusion_weight_gen(f_img, f_clin)
            alpha_clin, alpha_int = alphas[:, 0:1], alphas[:, 1:2]
        elif self.ablation == "E1-A2":
            f_miss = self.missing_mlp(R, r_global)
            alphas = self.fusion_weight_gen(f_miss, r_global)
            alpha_clin, alpha_int = alphas[:, 0:1], alphas[:, 1:2]
        elif self.ablation == "E1-A3":
            alpha_clin = torch.full((b, 1), 0.5, device=device, dtype=f_img.dtype)
            alpha_int = torch.full((b, 1), 0.5, device=device, dtype=f_img.dtype)
        elif self.ablation == "E1-C1":
            alpha_clin = torch.zeros((b, 1), device=device, dtype=f_img.dtype)
            alpha_int = torch.ones((b, 1), device=device, dtype=f_img.dtype)
        else:
            raise ValueError(f"Unknown ablation: {self.ablation}")
        if self.ablation not in ("E1-A3", "E1-C1"):
            scale = (alpha_clin + alpha_int).clamp(min=1e-8)
            alpha_clin = alpha_clin / scale
            alpha_int = alpha_int / scale
        f_fusion = alpha_clin * f_clin + alpha_int * f_int
        logits = self.classifier(f_fusion)
        aux = {
            "f_img": f_img,
            "f_clin": f_clin,
            "f_int": f_int,
            "f_fusion": f_fusion,
            "alpha_img": torch.zeros((b, 1), device=device, dtype=f_img.dtype),
            "alpha_clin": alpha_clin,
            "alpha_int": alpha_int,
        }
        return logits, aux


def build_ablation_model_proco(
    ablation: Optional[str],
    clin_dim: int,
    num_classes: int = 4,
    df: int = 256,
    clin_emb_dim: int = 256,
    f_miss_dim: int = 64,
) -> nn.Module:
    """构建 ProCo 版模型。E1-D1 / E1-D3 只改损失，结构仍用 E1 主模型。"""
    if ablation in ("E1", "E1-D1", "E1-D3"):
        return E1MainModel(
            clin_dim=clin_dim,
            num_classes=num_classes,
            df=df,
            clin_emb_dim=clin_emb_dim,
            f_miss_dim=f_miss_dim,
        )
    if ablation in ("E1-A1", "E1-A2", "E1-A3", "E1-C1"):
        return E1AblationModel(
            ablation=ablation,
            clin_dim=clin_dim,
            num_classes=num_classes,
            df=df,
            clin_emb_dim=clin_emb_dim,
            f_miss_dim=f_miss_dim,
        )
    if ablation is None or ablation == "full":
        return MissingAwareFusionModelProCo(
            clin_dim=clin_dim,
            num_classes=num_classes,
            df=df,
            clin_emb_dim=clin_emb_dim,
            f_miss_dim=f_miss_dim,
        )
    return AblationFusionModelProCo(
        ablation=ablation,
        clin_dim=clin_dim,
        num_classes=num_classes,
        df=df,
        clin_emb_dim=clin_emb_dim,
        f_miss_dim=f_miss_dim,
    )
