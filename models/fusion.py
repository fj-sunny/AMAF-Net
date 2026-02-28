"""
多模态融合模型封装

说明：
- Ours（Missing-aware Fusion）及消融 A1–E1 使用本包内部分解实现。
- DAFT 等基线仍从原脚本导入，待后续迁入本包后移除。
- 所有模型 forward 接口约定为：

    forward(x_ct, C_tilde, R, r_global) -> (logits, aux_dict)

  其中 aux_dict 至少包含键：
    - f_img, f_clin, f_int
    - alpha_img, alpha_clin, alpha_int
  以兼容现有评估与 ProCo 模块。
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from .missing_aware_fusion import (
    MissingAwareFusionModel,
    AblationFusionModel,
    build_ablation_model,
    ABLATION_NAMES,
    ABLATION_DESCRIPTIONS,
)

from train_missing_aware_fusion_proco import (  # type: ignore
    MissingAwareFusionModelProCo as _MissingAwareFusionModelProCo,
    AblationFusionModelProCo as _AblationFusionModelProCo,
    build_ablation_model_proco as _build_ablation_model_proco,
)

from train_missing_aware_fusion_daft import (  # type: ignore
    MissingAwareFusionModelDAFT as _MissingAwareFusionModelDAFT,
    MissingAwareHyperFCModelDAFT as _MissingAwareHyperFCModelDAFT,
    MissingAwareLateFusionModelDAFT as _MissingAwareLateFusionModelDAFT,
    MissingAwareCrossAttnFusionModelDAFT as _MissingAwareCrossAttnFusionModelDAFT,
    ResNetMLPConcatModel as _ResNetMLPConcatModel,
    ResNetMLPCrossAttnModel as _ResNetMLPCrossAttnModel,
    ResNetMLPDrFuseModel as _ResNetMLPDrFuseModel,
    build_ablation_model_daft as _build_ablation_model_daft,
)


# Ours 已由 .missing_aware_fusion 提供；ProCo/DAFT 仍来自原脚本
MissingAwareFusionModelProCo = _MissingAwareFusionModelProCo
AblationFusionModelProCo = _AblationFusionModelProCo
build_ablation_model_proco = _build_ablation_model_proco

MissingAwareFusionModelDAFT = _MissingAwareFusionModelDAFT
MissingAwareHyperFCModelDAFT = _MissingAwareHyperFCModelDAFT
MissingAwareLateFusionModelDAFT = _MissingAwareLateFusionModelDAFT
MissingAwareCrossAttnFusionModelDAFT = _MissingAwareCrossAttnFusionModelDAFT

ResNetMLPConcatModel = _ResNetMLPConcatModel
ResNetMLPCrossAttnModel = _ResNetMLPCrossAttnModel
ResNetMLPDrFuseModel = _ResNetMLPDrFuseModel


def build_daft_or_baseline(
    name: Optional[str],
    clin_dim: int,
    num_classes: int,
    df: int = 256,
    clin_emb_dim: int = 256,
    f_miss_dim: int = 64,
) -> nn.Module:
    """
    便捷封装 train_missing_aware_fusion_daft.build_ablation_model_daft，
    统一名称：
        - "daft" / "daft_full" -> MissingAwareFusionModelDAFT
        - "concat" -> ResNetMLPConcatModel
        - "cross_attn" -> ResNetMLPCrossAttnModel
        - "late" -> MissingAwareLateFusionModelDAFT
        - "hyperfusion" -> MissingAwareHyperFCModelDAFT
        - "drfuse" -> ResNetMLPDrFuseModel
        - 其他 A1/A2/... 消融则直接下沉给原函数
    """
    if name is None or name in ("full", "daft", "daft_full"):
        return MissingAwareFusionModelDAFT(
            clin_dim=clin_dim,
            num_classes=num_classes,
            df=df,
            clin_emb_dim=clin_emb_dim,
            f_miss_dim=f_miss_dim,
        )

    lower = name.lower()
    if lower in ("concat", "cat", "resnet_mlp_concat"):
        return ResNetMLPConcatModel(clin_dim=clin_dim, num_classes=num_classes, hidden=df)
    if lower in ("cross_attn", "crossattention", "ca"):
        return ResNetMLPCrossAttnModel(clin_dim=clin_dim, num_classes=num_classes, hidden=df)
    if lower in ("late", "latefusion", "lf"):
        return MissingAwareLateFusionModelDAFT(
            clin_dim=clin_dim,
            num_classes=num_classes,
            df=df,
            clin_emb_dim=clin_emb_dim,
            f_miss_dim=f_miss_dim,
        )
    if lower in ("hyperfc", "hyperfusion", "hf"):
        return MissingAwareHyperFCModelDAFT(
            clin_dim=clin_dim,
            num_classes=num_classes,
            df=df,
            clin_emb_dim=clin_emb_dim,
            f_miss_dim=f_miss_dim,
        )
    if lower in ("drfuse", "dr"):
        return ResNetMLPDrFuseModel(clin_dim=clin_dim, num_classes=num_classes, hidden=df)

    # 其余名称（如 A1/A2/B1/...）直接交给原构造函数
    return _build_ablation_model_daft(
        ablation=name,
        clin_dim=clin_dim,
        num_classes=num_classes,
        df=df,
        clin_emb_dim=clin_emb_dim,
        f_miss_dim=f_miss_dim,
    )


