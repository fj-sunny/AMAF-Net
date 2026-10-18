# -*- coding: utf-8 -*-
"""DAFT 融合与对照基线。

结构来自 train_missing_aware_fusion_daft.py，骨干与临床编码器使用本包实现，
不再加载外部脚本。
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet3d import BasicBlock3D, ResNet18_3D_Encoder
from .clinical_encoders import (
    ClinicalEncoderCtildeOnly,
    ClinicalEncoderMissingAware,
    MissingPatternMLP,
    MissingPatternMLP_RGlobalOnly,
)
from .fusion_weights import (
    FusionWeightGenerator,
    FusionWeightGenerator2,
    FusionWeightGeneratorFromFeat,
)

C_LAYER4 = 512


class DAFTConditionModule(nn.Module):
    """u = [s; f_clin; f_miss; r_global] -> [γ, β]。γ = 1 + α·tanh(γ_raw)。"""

    def __init__(self, u_dim: int, num_channels: int = C_LAYER4, bottleneck_dim: int = 128):
        super().__init__()
        self.num_channels = num_channels
        self.mlp = nn.Sequential(
            nn.Linear(u_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, 2 * num_channels),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        if self.mlp[-1].bias is not None:
            nn.init.zeros_(self.mlp[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.beta_param = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.mlp(u)
        gamma_raw, beta_raw = out.chunk(2, dim=1)
        gamma = 1.0 + self.alpha * torch.tanh(gamma_raw)
        beta = self.beta_param * torch.tanh(beta_raw)
        return gamma, beta


class ResNet18_3D_DAFT_Encoder(nn.Module):
    """layer4 特征经 DAFT 调制后 GAP，输出 (B, 512)。"""

    def __init__(
        self,
        clin_emb_dim: int,
        f_miss_dim: int,
        in_channels: int = 1,
        daft_bottleneck_dim: int = 128,
    ):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock3D, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock3D, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock3D, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock3D, 512, 2, stride=2)
        self.gap = nn.AdaptiveAvgPool3d(1)
        u_dim = C_LAYER4 + clin_emb_dim + f_miss_dim + 1
        self.daft_h = DAFTConditionModule(
            u_dim=u_dim, num_channels=C_LAYER4, bottleneck_dim=daft_bottleneck_dim
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = [
            block(self.inplanes, planes, stride=stride, downsample=downsample, use_attention=False, attn_type="SE")
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_attention=False, attn_type="SE"))
        return nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        f_clin: torch.Tensor,
        f_miss: torch.Tensor,
        r_global: torch.Tensor,
    ) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat = self.layer4(x)
        s = self.gap(feat).flatten(1)
        if r_global.dim() == 1:
            r_global = r_global.unsqueeze(1)
        u = torch.cat([s, f_clin, f_miss, r_global], dim=1)
        gamma, beta = self.daft_h(u)
        gamma = gamma.view(*gamma.shape, 1, 1, 1).expand_as(feat)
        beta = beta.view(*beta.shape, 1, 1, 1).expand_as(feat)
        y = gamma * feat + beta
        return self.gap(y).flatten(1)


class HyperFC(nn.Module):
    """由临床 embedding 生成按样本变化的线性分类头 (W, b)。"""

    def __init__(self, emb_dim: int, in_dim: int, out_dim: int, hidden: int = 128, init_std: float = 1e-3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.encoder = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.to_w = nn.Linear(hidden, out_dim * in_dim)
        self.to_b = nn.Linear(hidden, out_dim)
        nn.init.normal_(self.to_w.weight, mean=0.0, std=init_std)
        nn.init.constant_(self.to_w.bias, 0.0)
        nn.init.normal_(self.to_b.weight, mean=0.0, std=init_std)
        nn.init.constant_(self.to_b.bias, 0.0)

    def forward(self, clin_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(clin_emb)
        w = self.to_w(h).view(-1, self.out_dim, self.in_dim)
        b = self.to_b(h)
        return w, b


class CrossAttentionFusion(nn.Module):
    """单头交叉注意力，query/key/value 均为 (B, D)。"""

    def __init__(self, query_dim: int, key_dim: int, value_dim: int, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(query_dim, d_model)
        self.W_k = nn.Linear(key_dim, d_model)
        self.W_v = nn.Linear(value_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        q = self.W_q(query).unsqueeze(1)
        k = self.W_k(key).unsqueeze(1)
        v = self.W_v(value).unsqueeze(1)
        scale = self.d_model ** -0.5
        attn = F.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        attn = self.dropout(attn)
        return (attn @ v).squeeze(1)


def _two_branch_aux(f_img, f_clin, alpha_img, alpha_clin):
    b = f_img.size(0)
    return {
        "f_img": f_img,
        "f_clin": f_clin,
        "alpha_img": alpha_img,
        "alpha_clin": alpha_clin,
        "f_int": f_img,
        "alpha_int": torch.zeros((b, 1), device=f_img.device, dtype=f_img.dtype),
        "f_fusion": alpha_img * f_img + alpha_clin * f_clin,
    }


class MissingAwareFusionModelDAFT(nn.Module):
    """DAFT 影像分支 + 缺失感知两路加权：f = α_img·f_img + α_clin·f_clin。"""

    def __init__(
        self,
        clin_dim: int,
        num_classes: int = 4,
        df: int = 256,
        clin_emb_dim: int = 256,
        f_miss_dim: int = 64,
        daft_bottleneck_dim: int = 128,
    ):
        super().__init__()
        self.df = df
        self.clin_emb_dim = clin_emb_dim
        self.ct_encoder_daft = ResNet18_3D_DAFT_Encoder(
            clin_emb_dim=clin_emb_dim,
            f_miss_dim=f_miss_dim,
            in_channels=1,
            daft_bottleneck_dim=daft_bottleneck_dim,
        )
        self.ct_proj = nn.Linear(512, df)
        self.missing_mlp = MissingPatternMLP(d=clin_dim, out_dim=f_miss_dim, hidden=(128, 64), dropout=0.2)
        self.clin_encoder = ClinicalEncoderMissingAware(
            d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2
        )
        self.fusion_weight_gen = FusionWeightGenerator2(
            f_miss_dim=f_miss_dim, num_weights=2, hidden=64, dropout=0.1
        )
        self.classifier = nn.Linear(df, num_classes)

    def forward(self, x_ct, C_tilde, R, r_global):
        f_clin = self.clin_encoder(C_tilde, R)
        f_miss = self.missing_mlp(R, r_global)
        f_img = self.ct_proj(self.ct_encoder_daft(x_ct, f_clin, f_miss, r_global))
        alphas = self.fusion_weight_gen(f_miss, r_global)
        alpha_img, alpha_clin = alphas[:, 0:1], alphas[:, 1:2]
        f = alpha_img * f_img + alpha_clin * f_clin
        logits = self.classifier(f)
        return logits, _two_branch_aux(f_img, f_clin, alpha_img, alpha_clin)


class MissingAwareHyperFCModelDAFT(nn.Module):
    """特征级缺失感知融合，再加上由临床 embedding 生成的样本自适应分类头。"""

    def __init__(
        self,
        clin_dim: int,
        num_classes: int = 4,
        df: int = 256,
        clin_emb_dim: int = 256,
        f_miss_dim: int = 64,
        daft_bottleneck_dim: int = 128,
        hyper_hidden: int = 128,
        hyper_init_std: float = 1e-3,
    ):
        super().__init__()
        self.df = df
        self.clin_emb_dim = clin_emb_dim
        self.ct_encoder_daft = ResNet18_3D_DAFT_Encoder(
            clin_emb_dim=clin_emb_dim,
            f_miss_dim=f_miss_dim,
            in_channels=1,
            daft_bottleneck_dim=daft_bottleneck_dim,
        )
        self.ct_proj = nn.Linear(512, df)
        self.missing_mlp = MissingPatternMLP(d=clin_dim, out_dim=f_miss_dim, hidden=(128, 64), dropout=0.2)
        self.clin_encoder = ClinicalEncoderMissingAware(
            d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2
        )
        self.fusion_weight_gen = FusionWeightGenerator2(
            f_miss_dim=f_miss_dim, num_weights=2, hidden=64, dropout=0.1
        )
        self.classifier = nn.Linear(df, num_classes)
        self.ct_ln = nn.LayerNorm(df)
        self.clin_ln = nn.LayerNorm(clin_emb_dim)
        self.clin_scale = nn.Parameter(torch.tensor(1.0))
        self.hyper = HyperFC(
            emb_dim=clin_emb_dim,
            in_dim=df,
            out_dim=num_classes,
            hidden=hyper_hidden,
            init_std=hyper_init_std,
        )

    def forward(self, x_ct, C_tilde, R, r_global):
        f_clin = self.clin_encoder(C_tilde, R)
        f_miss = self.missing_mlp(R, r_global)
        f_img = self.ct_proj(self.ct_encoder_daft(x_ct, f_clin, f_miss, r_global))
        alphas = self.fusion_weight_gen(f_miss, r_global)
        alpha_img, alpha_clin = alphas[:, 0:1], alphas[:, 1:2]
        f_fusion = alpha_img * f_img + alpha_clin * f_clin
        logits_base = self.classifier(f_fusion)
        ct_feat = self.ct_ln(f_img)
        clin_emb = self.clin_ln(f_clin) * self.clin_scale
        w, b = self.hyper(clin_emb)
        hyper_logits = torch.bmm(w, ct_feat.unsqueeze(-1)).squeeze(-1) + b
        logits = logits_base + hyper_logits
        aux = _two_branch_aux(f_img, f_clin, alpha_img, alpha_clin)
        aux["f_fusion"] = f_fusion
        return logits, aux


class MissingAwareLateFusionModelDAFT(nn.Module):
    """在 logits 上按缺失感知权重做晚期融合。"""

    def __init__(
        self,
        clin_dim: int,
        num_classes: int = 4,
        df: int = 256,
        clin_emb_dim: int = 256,
        f_miss_dim: int = 64,
        daft_bottleneck_dim: int = 128,
    ):
        super().__init__()
        self.df = df
        self.clin_emb_dim = clin_emb_dim
        self.ct_encoder_daft = ResNet18_3D_DAFT_Encoder(
            clin_emb_dim=clin_emb_dim,
            f_miss_dim=f_miss_dim,
            in_channels=1,
            daft_bottleneck_dim=daft_bottleneck_dim,
        )
        self.ct_proj = nn.Linear(512, df)
        self.missing_mlp = MissingPatternMLP(d=clin_dim, out_dim=f_miss_dim, hidden=(128, 64), dropout=0.2)
        self.clin_encoder = ClinicalEncoderMissingAware(
            d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2
        )
        self.fusion_weight_gen = FusionWeightGenerator2(
            f_miss_dim=f_miss_dim, num_weights=2, hidden=64, dropout=0.1
        )
        self.classifier_img = nn.Linear(df, num_classes)
        self.classifier_clin = nn.Linear(clin_emb_dim, num_classes)

    def forward(self, x_ct, C_tilde, R, r_global):
        f_clin = self.clin_encoder(C_tilde, R)
        f_miss = self.missing_mlp(R, r_global)
        f_img = self.ct_proj(self.ct_encoder_daft(x_ct, f_clin, f_miss, r_global))
        alphas = self.fusion_weight_gen(f_miss, r_global)
        alpha_img, alpha_clin = alphas[:, 0:1], alphas[:, 1:2]
        logits = alpha_img * self.classifier_img(f_img) + alpha_clin * self.classifier_clin(f_clin)
        return logits, _two_branch_aux(f_img, f_clin, alpha_img, alpha_clin)


class MissingAwareCrossAttnFusionModelDAFT(nn.Module):
    """DAFT 特征经双向交叉注意力后再做缺失感知加权。"""

    def __init__(
        self,
        clin_dim: int,
        num_classes: int = 4,
        df: int = 256,
        clin_emb_dim: int = 256,
        f_miss_dim: int = 64,
        daft_bottleneck_dim: int = 128,
        cross_attn_dim: int = 256,
        cross_attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.df = df
        self.clin_emb_dim = clin_emb_dim
        self.ct_encoder_daft = ResNet18_3D_DAFT_Encoder(
            clin_emb_dim=clin_emb_dim,
            f_miss_dim=f_miss_dim,
            in_channels=1,
            daft_bottleneck_dim=daft_bottleneck_dim,
        )
        self.ct_proj = nn.Linear(512, df)
        self.missing_mlp = MissingPatternMLP(d=clin_dim, out_dim=f_miss_dim, hidden=(128, 64), dropout=0.2)
        self.clin_encoder = ClinicalEncoderMissingAware(
            d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2
        )
        self.fusion_weight_gen = FusionWeightGenerator2(
            f_miss_dim=f_miss_dim, num_weights=2, hidden=64, dropout=0.1
        )
        self.cross_attn_img2clin = CrossAttentionFusion(
            query_dim=df, key_dim=clin_emb_dim, value_dim=clin_emb_dim,
            d_model=cross_attn_dim, dropout=cross_attn_dropout,
        )
        self.cross_attn_clin2img = CrossAttentionFusion(
            query_dim=clin_emb_dim, key_dim=df, value_dim=df,
            d_model=cross_attn_dim, dropout=cross_attn_dropout,
        )
        self.proj_img_ctx = nn.Linear(cross_attn_dim, df)
        self.proj_clin_ctx = nn.Linear(cross_attn_dim, clin_emb_dim)
        self.classifier = nn.Linear(df, num_classes)

    def forward(self, x_ct, C_tilde, R, r_global):
        f_clin = self.clin_encoder(C_tilde, R)
        f_miss = self.missing_mlp(R, r_global)
        f_img = self.ct_proj(self.ct_encoder_daft(x_ct, f_clin, f_miss, r_global))
        img_ctx = self.cross_attn_img2clin(query=f_img, key=f_clin, value=f_clin)
        clin_ctx = self.cross_attn_clin2img(query=f_clin, key=f_img, value=f_img)
        f_img_enh = f_img + self.proj_img_ctx(img_ctx)
        f_clin_enh = f_clin + self.proj_clin_ctx(clin_ctx)
        alphas = self.fusion_weight_gen(f_miss, r_global)
        alpha_img, alpha_clin = alphas[:, 0:1], alphas[:, 1:2]
        f = alpha_img * f_img_enh + alpha_clin * f_clin_enh
        logits = self.classifier(f)
        aux = _two_branch_aux(f_img, f_clin, alpha_img, alpha_clin)
        aux["f_fusion"] = f
        return logits, aux


class ResNetMLPConcatModel(nn.Module):
    """ResNet18 + 临床 MLP，拼接后分类。"""

    def __init__(self, clin_dim: int, num_classes: int = 4, hidden: int = 256):
        super().__init__()
        self.df = hidden
        self.hidden = hidden
        self.ct_encoder = ResNet18_3D_Encoder(in_channels=1)
        self.ct_proj = nn.Linear(512, hidden)
        self.clin_encoder = ClinicalEncoderCtildeOnly(
            d=clin_dim, out_dim=hidden, hidden=(256, 128), dropout=0.2
        )
        self.classifier = nn.Linear(hidden * 2, num_classes)

    def forward(self, x_ct, C_tilde, R, r_global):
        b = x_ct.size(0)
        f_img = self.ct_proj(self.ct_encoder(x_ct))
        f_clin = self.clin_encoder(C_tilde, R)
        logits = self.classifier(torch.cat([f_img, f_clin], dim=1))
        aux = {
            "f_img": f_img,
            "f_clin": f_clin,
            "alpha_img": torch.ones((b, 1), device=f_img.device, dtype=f_img.dtype),
            "alpha_clin": torch.zeros((b, 1), device=f_img.device, dtype=f_img.dtype),
            "f_int": f_img,
            "alpha_int": torch.zeros((b, 1), device=f_img.device, dtype=f_img.dtype),
            "f_fusion": torch.cat([f_img, f_clin], dim=1),
        }
        return logits, aux


class ResNetMLPCrossAttnModel(nn.Module):
    """ResNet18 + 临床 MLP + 双向交叉注意力，不使用 DAFT。"""

    def __init__(
        self,
        clin_dim: int,
        num_classes: int = 4,
        hidden: int = 256,
        cross_attn_dim: int = 256,
        cross_attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.df = hidden
        self.hidden = hidden
        self.ct_encoder = ResNet18_3D_Encoder(in_channels=1)
        self.ct_proj = nn.Linear(512, hidden)
        self.clin_encoder = ClinicalEncoderCtildeOnly(
            d=clin_dim, out_dim=hidden, hidden=(256, 128), dropout=0.2
        )
        self.cross_attn_img2clin = CrossAttentionFusion(
            query_dim=hidden, key_dim=hidden, value_dim=hidden,
            d_model=cross_attn_dim, dropout=cross_attn_dropout,
        )
        self.cross_attn_clin2img = CrossAttentionFusion(
            query_dim=hidden, key_dim=hidden, value_dim=hidden,
            d_model=cross_attn_dim, dropout=cross_attn_dropout,
        )
        self.proj_img_ctx = nn.Linear(cross_attn_dim, hidden)
        self.proj_clin_ctx = nn.Linear(cross_attn_dim, hidden)
        self.classifier = nn.Linear(hidden * 2, num_classes)

    def forward(self, x_ct, C_tilde, R, r_global):
        b = x_ct.size(0)
        f_img = self.ct_proj(self.ct_encoder(x_ct))
        f_clin = self.clin_encoder(C_tilde, R)
        img_ctx = self.cross_attn_img2clin(query=f_img, key=f_clin, value=f_clin)
        clin_ctx = self.cross_attn_clin2img(query=f_clin, key=f_img, value=f_img)
        f_img_enh = f_img + self.proj_img_ctx(img_ctx)
        f_clin_enh = f_clin + self.proj_clin_ctx(clin_ctx)
        logits = self.classifier(torch.cat([f_img_enh, f_clin_enh], dim=1))
        aux = {
            "f_img": f_img,
            "f_clin": f_clin,
            "alpha_img": torch.ones((b, 1), device=f_img.device, dtype=f_img.dtype),
            "alpha_clin": torch.zeros((b, 1), device=f_img.device, dtype=f_img.dtype),
            "f_int": f_img,
            "alpha_int": torch.zeros((b, 1), device=f_img.device, dtype=f_img.dtype),
            "f_fusion": torch.cat([f_img_enh, f_clin_enh], dim=1),
        }
        return logits, aux


class ResNetMLPDrFuseModel(nn.Module):
    """ResNet18 + 临床 MLP 的 DrFuse 式共享/特有特征融合。"""

    def __init__(self, clin_dim: int, num_classes: int = 4, hidden: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.hidden = hidden
        self.df = hidden
        self.ct_encoder = ResNet18_3D_Encoder(in_channels=1)
        self.ct_proj = nn.Linear(512, hidden)
        self.clin_encoder = ClinicalEncoderCtildeOnly(
            d=clin_dim, out_dim=hidden, hidden=(256, 128), dropout=0.2
        )
        self.shared_project = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.ehr_dist_proj = nn.Linear(hidden, hidden)
        self.cxr_dist_proj = nn.Linear(hidden, hidden)
        self.attn_proj = nn.Linear(hidden, (2 + num_classes) * hidden)
        self.final_pred_fc = nn.Linear(hidden, num_classes)

    def forward(self, x_ct, C_tilde, R, r_global):
        b = x_ct.size(0)
        device = x_ct.device
        ct_feat = self.ct_proj(self.ct_encoder(x_ct))
        clin_feat = self.clin_encoder(C_tilde, R)
        feat_ehr_shared = self.shared_project(clin_feat)
        feat_cxr_shared = self.shared_project(ct_feat)
        feat_ehr_distinct = self.ehr_dist_proj(clin_feat)
        feat_cxr_distinct = self.cxr_dist_proj(ct_feat)
        pairs = torch.ones(b, 1, device=device, dtype=ct_feat.dtype)
        h1, h2 = feat_ehr_shared, feat_cxr_shared
        term1 = torch.stack([h1 + h2, h1 + h2, h1, h2], dim=2)
        term2 = torch.stack([torch.zeros_like(h1), torch.zeros_like(h1), h1, h2], dim=2)
        feat_avg_shared = torch.logsumexp(term1, dim=2) - torch.logsumexp(term2, dim=2)
        feat_avg_shared = pairs * feat_avg_shared + (1 - pairs) * feat_ehr_shared
        attn_input = torch.stack([feat_ehr_distinct, feat_avg_shared, feat_cxr_distinct], dim=1)
        qkvs = self.attn_proj(attn_input)
        q, v, *k = qkvs.chunk(2 + self.num_classes, dim=-1)
        q_mean = q.mean(dim=1)
        ks = torch.stack(k, dim=1)
        attn_logits = torch.einsum("bd,bnkd->bnk", q_mean, ks) / (q.shape[-1] ** 0.5)
        attn_weights = F.softmax(attn_logits, dim=-1)
        feat_final = torch.matmul(attn_weights, v)
        pred_final = self.final_pred_fc(feat_final)
        logits = torch.diagonal(pred_final, dim1=1, dim2=2)
        alpha_img = torch.ones((b, 1), device=device, dtype=feat_avg_shared.dtype)
        alpha_zero = torch.zeros((b, 1), device=device, dtype=feat_avg_shared.dtype)
        aux = {
            "f_img": feat_avg_shared,
            "f_clin": torch.zeros_like(feat_avg_shared),
            "f_int": torch.zeros_like(feat_avg_shared),
            "alpha_img": alpha_img,
            "alpha_clin": alpha_zero,
            "alpha_int": alpha_zero,
            "f_fusion": feat_avg_shared,
            "feat_ehr_shared": feat_ehr_shared,
            "feat_cxr_shared": feat_cxr_shared,
            "feat_ehr_distinct": feat_ehr_distinct,
            "feat_cxr_distinct": feat_cxr_distinct,
            "feat_avg_shared": feat_avg_shared,
        }
        return logits, aux


class AblationFusionModelDAFT(nn.Module):
    """DAFT 消融：融合只在 f_img 与 f_clin 之间。B1 去掉 DAFT，改用纯 ResNet。"""

    def __init__(
        self,
        ablation: str,
        clin_dim: int,
        num_classes: int = 4,
        df: int = 256,
        clin_emb_dim: int = 256,
        f_miss_dim: int = 64,
        daft_bottleneck_dim: int = 128,
    ):
        super().__init__()
        self.ablation = ablation
        self.df = df
        self.clin_emb_dim = clin_emb_dim
        self.f_miss_dim = f_miss_dim
        use_daft = True
        if ablation == "A1":
            self.clin_encoder = ClinicalEncoderCtildeOnly(
                d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2
            )
            self.missing_mlp = None
            self.fusion_weight_gen = FusionWeightGeneratorFromFeat(
                feat_dim=df + clin_emb_dim, num_weights=2, hidden=64, dropout=0.1
            )
            self.use_alpha_clin = True
            self.fixed_alphas = None
        elif ablation == "A2":
            self.clin_encoder = ClinicalEncoderMissingAware(
                d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2
            )
            self.missing_mlp = MissingPatternMLP_RGlobalOnly(out_dim=f_miss_dim, hidden=(64, 32), dropout=0.2)
            self.fusion_weight_gen = FusionWeightGenerator(
                f_miss_dim=f_miss_dim, num_weights=2, hidden=64, dropout=0.1
            )
            self.use_alpha_clin = True
            self.fixed_alphas = None
        elif ablation == "A3":
            self.clin_encoder = ClinicalEncoderMissingAware(
                d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2
            )
            self.missing_mlp = MissingPatternMLP(d=clin_dim, out_dim=f_miss_dim, hidden=(128, 64), dropout=0.2)
            self.fusion_weight_gen = None
            self.use_alpha_clin = True
            self.fixed_alphas = (0.5, 0.5)
        elif ablation == "B1":
            self.clin_encoder = ClinicalEncoderMissingAware(
                d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2
            )
            self.missing_mlp = MissingPatternMLP(d=clin_dim, out_dim=f_miss_dim, hidden=(128, 64), dropout=0.2)
            self.fusion_weight_gen = FusionWeightGenerator2(
                f_miss_dim=f_miss_dim, num_weights=2, hidden=64, dropout=0.1
            )
            use_daft = False
            self.use_alpha_clin = True
            self.fixed_alphas = None
        elif ablation == "B2":
            self.clin_encoder = ClinicalEncoderMissingAware(
                d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2
            )
            self.missing_mlp = MissingPatternMLP(d=clin_dim, out_dim=f_miss_dim, hidden=(128, 64), dropout=0.2)
            self.fusion_weight_gen = FusionWeightGenerator2(
                f_miss_dim=f_miss_dim, num_weights=2, hidden=64, dropout=0.1
            )
            self.use_alpha_clin = True
            self.fixed_alphas = None
        elif ablation == "C1":
            self.clin_encoder = ClinicalEncoderMissingAware(
                d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2
            )
            self.missing_mlp = MissingPatternMLP(d=clin_dim, out_dim=f_miss_dim, hidden=(128, 64), dropout=0.2)
            self.fusion_weight_gen = FusionWeightGenerator2(
                f_miss_dim=f_miss_dim, num_weights=2, hidden=64, dropout=0.1
            )
            self.use_alpha_clin = False
            self.fixed_alphas = None
        elif ablation == "C2":
            self.clin_encoder = ClinicalEncoderMissingAware(
                d=clin_dim, out_dim=clin_emb_dim, hidden=(256, 128), dropout=0.2
            )
            self.missing_mlp = None
            self.fusion_weight_gen = None
            use_daft = False
            self.use_alpha_clin = True
            self.fixed_alphas = None
            self.fusion_concat = nn.Sequential(
                nn.Linear(df + clin_emb_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )
            self.classifier = None
        else:
            raise ValueError(f"Unknown ablation: {ablation}")

        self.use_daft = use_daft
        if use_daft:
            self.ct_encoder_daft = ResNet18_3D_DAFT_Encoder(
                clin_emb_dim=clin_emb_dim,
                f_miss_dim=f_miss_dim,
                in_channels=1,
                daft_bottleneck_dim=daft_bottleneck_dim,
            )
            self.ct_encoder = None
            self.ct_proj = nn.Linear(512, df)
        else:
            self.ct_encoder_daft = None
            self.ct_encoder = ResNet18_3D_Encoder(in_channels=1)
            self.ct_proj = nn.Linear(512, df)
        if ablation != "C2":
            self.classifier = nn.Linear(df, num_classes)
            self.fusion_concat = None

    def _get_f_miss_for_daft(self, R, r_global):
        if self.missing_mlp is not None:
            return self.missing_mlp(R, r_global)
        return torch.zeros(R.size(0), self.f_miss_dim, device=R.device, dtype=R.dtype)

    def forward(self, x_ct, C_tilde, R, r_global):
        b = x_ct.size(0)
        device = x_ct.device
        f_clin = self.clin_encoder(C_tilde, R)
        if self.ct_encoder_daft is not None:
            f_miss_daft = self._get_f_miss_for_daft(R, r_global)
            f_img = self.ct_proj(self.ct_encoder_daft(x_ct, f_clin, f_miss_daft, r_global))
        else:
            f_img = self.ct_proj(self.ct_encoder(x_ct))
        if self.ablation == "C2":
            fused = torch.cat([f_img, f_clin], dim=1)
            logits = self.fusion_concat(fused)
            aux = {
                "f_img": f_img,
                "f_clin": f_clin,
                "alpha_img": torch.full((b, 1), 0.5, device=device, dtype=f_img.dtype),
                "alpha_clin": torch.full((b, 1), 0.5, device=device, dtype=f_img.dtype),
                "f_int": torch.zeros_like(f_img),
                "alpha_int": torch.zeros((b, 1), device=device, dtype=f_img.dtype),
                "f_fusion": fused,
            }
            return logits, aux
        if self.fixed_alphas is not None:
            a1, a2 = self.fixed_alphas
            alpha_img = torch.full((b, 1), a1, device=device, dtype=f_img.dtype)
            alpha_clin = torch.full((b, 1), a2, device=device, dtype=f_img.dtype)
        elif self.ablation == "A1":
            alphas = self.fusion_weight_gen(f_img, f_clin)
            alpha_img, alpha_clin = alphas[:, 0:1], alphas[:, 1:2]
        else:
            f_miss = self.missing_mlp(R, r_global)
            alphas = self.fusion_weight_gen(f_miss, r_global)
            alpha_img, alpha_clin = alphas[:, 0:1], alphas[:, 1:2]
            if not self.use_alpha_clin:
                alpha_clin = torch.zeros((b, 1), device=device, dtype=f_img.dtype)
        if self.use_alpha_clin:
            f = alpha_img * f_img + alpha_clin * f_clin
        else:
            f = alpha_img * f_img
            alpha_clin = torch.zeros((b, 1), device=device, dtype=f_img.dtype)
        logits = self.classifier(f)
        aux = _two_branch_aux(f_img, f_clin, alpha_img, alpha_clin)
        aux["f_fusion"] = f
        return logits, aux


def build_ablation_model_daft(
    ablation: Optional[str],
    clin_dim: int,
    num_classes: int = 4,
    df: int = 256,
    clin_emb_dim: int = 256,
    f_miss_dim: int = 64,
    daft_bottleneck_dim: int = 128,
) -> nn.Module:
    """按名称构建 DAFT 完整模型、对照基线或消融。"""
    if ablation is None or ablation in ("full", "daft", "daft_full"):
        return MissingAwareFusionModelDAFT(
            clin_dim=clin_dim,
            num_classes=num_classes,
            df=df,
            clin_emb_dim=clin_emb_dim,
            f_miss_dim=f_miss_dim,
            daft_bottleneck_dim=daft_bottleneck_dim,
        )
    name = str(ablation).lower()
    if name in ("concat", "cat", "resnet_mlp_concat"):
        return ResNetMLPConcatModel(clin_dim=clin_dim, num_classes=num_classes, hidden=df)
    if name in ("drfuse", "dr"):
        return ResNetMLPDrFuseModel(clin_dim=clin_dim, num_classes=num_classes, hidden=df)
    if name in ("hyperfc", "hyperfusion", "hf"):
        return MissingAwareHyperFCModelDAFT(
            clin_dim=clin_dim,
            num_classes=num_classes,
            df=df,
            clin_emb_dim=clin_emb_dim,
            f_miss_dim=f_miss_dim,
            daft_bottleneck_dim=daft_bottleneck_dim,
        )
    if name in ("late", "latefusion", "lf"):
        return MissingAwareLateFusionModelDAFT(
            clin_dim=clin_dim,
            num_classes=num_classes,
            df=df,
            clin_emb_dim=clin_emb_dim,
            f_miss_dim=f_miss_dim,
            daft_bottleneck_dim=daft_bottleneck_dim,
        )
    if name in ("cross_attn", "crossattention", "ca"):
        return ResNetMLPCrossAttnModel(clin_dim=clin_dim, num_classes=num_classes, hidden=df)
    if name in ("daft_ca", "daft_cross_attn"):
        return MissingAwareCrossAttnFusionModelDAFT(
            clin_dim=clin_dim,
            num_classes=num_classes,
            df=df,
            clin_emb_dim=clin_emb_dim,
            f_miss_dim=f_miss_dim,
            daft_bottleneck_dim=daft_bottleneck_dim,
        )
    return AblationFusionModelDAFT(
        ablation=ablation,
        clin_dim=clin_dim,
        num_classes=num_classes,
        df=df,
        clin_emb_dim=clin_emb_dim,
        f_miss_dim=f_miss_dim,
        daft_bottleneck_dim=daft_bottleneck_dim,
    )
