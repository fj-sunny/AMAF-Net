"""
多模态融合模型。

- Ours 与 A1–E1：models.missing_aware_fusion
- Ours + ProCo、E1 系列：models.proco
- DAFT / Concat / Cross-Attn / Late / HyperFusion / DrFuse：models.daft

forward(x_ct, C_tilde, R, r_global) -> (logits, aux)
aux 至少包含 f_img, f_clin, f_int, alpha_img, alpha_clin, alpha_int。
ProCo 训练额外使用 f_fusion。
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from .daft import (
    AblationFusionModelDAFT,
    MissingAwareCrossAttnFusionModelDAFT,
    MissingAwareFusionModelDAFT,
    MissingAwareHyperFCModelDAFT,
    MissingAwareLateFusionModelDAFT,
    ResNetMLPConcatModel,
    ResNetMLPCrossAttnModel,
    ResNetMLPDrFuseModel,
    build_ablation_model_daft,
)
from .missing_aware_fusion import (
    ABLATION_DESCRIPTIONS,
    ABLATION_NAMES,
    AblationFusionModel,
    MissingAwareFusionModel,
    build_ablation_model,
)
from .proco import (
    AblationFusionModelProCo,
    E1AblationModel,
    E1MainModel,
    MissingAwareFusionModelProCo,
    build_ablation_model_proco,
)


def build_daft_or_baseline(
    name: Optional[str],
    clin_dim: int,
    num_classes: int,
    df: int = 256,
    clin_emb_dim: int = 256,
    f_miss_dim: int = 64,
) -> nn.Module:
    """统一名称入口，实际构造交给 build_ablation_model_daft。"""
    return build_ablation_model_daft(
        ablation=name,
        clin_dim=clin_dim,
        num_classes=num_classes,
        df=df,
        clin_emb_dim=clin_emb_dim,
        f_miss_dim=f_miss_dim,
    )


__all__ = [
    "ABLATION_NAMES",
    "ABLATION_DESCRIPTIONS",
    "MissingAwareFusionModel",
    "AblationFusionModel",
    "build_ablation_model",
    "MissingAwareFusionModelProCo",
    "AblationFusionModelProCo",
    "E1MainModel",
    "E1AblationModel",
    "build_ablation_model_proco",
    "MissingAwareFusionModelDAFT",
    "MissingAwareHyperFCModelDAFT",
    "MissingAwareLateFusionModelDAFT",
    "MissingAwareCrossAttnFusionModelDAFT",
    "AblationFusionModelDAFT",
    "ResNetMLPConcatModel",
    "ResNetMLPCrossAttnModel",
    "ResNetMLPDrFuseModel",
    "build_ablation_model_daft",
    "build_daft_or_baseline",
]
