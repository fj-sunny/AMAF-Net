"""
模型注册与统一构建接口

本模块将论文实验中提到的所有模型/变体统一注册为字符串 key，便于在主程序中通过
--model 或 --experiment 参数进行选择：

- 图像-only:
    - image_resnet18
    - image_unet3d
    - image_vit
- 临床-only:
    - clinical_mlp
    - clinical_mlp_missing
    - clinical_ft_transformer
- 融合基线:
    - fusion_concat
    - fusion_cross_attn
    - fusion_late
    - fusion_daft
    - fusion_hyperfusion
    - fusion_drfuse
- 缺失感知融合（Ours）及消融:
    - ours            (MissingAwareFusionModel, 不含 ProCo)
    - ours_proco      (MissingAwareFusionModelProCo + ProCo 训练)
    - ours_A1/A2/A3/B1/B2/C1/C2/E1 等（与原 ABLATION_NAMES 对应）
    - ours_E1_*       （E1 系列消融，可在 ProCo 脚本中选择）
"""

from __future__ import annotations

from typing import Callable, Dict, List

import torch.nn as nn

from . import backbones
from . import fusion


ModelBuilder = Callable[[int, int, int, int, int], nn.Module]
# 约定 builder 签名：
#   builder(clin_dim, num_classes, df, clin_emb_dim, f_miss_dim) -> nn.Module


def _wrap_image_only(builder_fn: Callable[..., nn.Module]) -> ModelBuilder:
    """将仅依赖 num_classes 的图像-only backbone 包装为统一签名。"""

    def _builder(clin_dim: int, num_classes: int, df: int, clin_emb_dim: int, f_miss_dim: int) -> nn.Module:
        return builder_fn(num_classes=num_classes)

    return _builder


def _wrap_clinical_only(builder_fn: Callable[..., nn.Module]) -> ModelBuilder:
    """将仅依赖 clin_dim/num_classes 的临床-only backbone 包装为统一签名。"""

    def _builder(clin_dim: int, num_classes: int, df: int, clin_emb_dim: int, f_miss_dim: int) -> nn.Module:
        return builder_fn(clin_dim=clin_dim, num_classes=num_classes)

    return _builder


def _wrap_fusion(builder_fn: Callable[..., nn.Module]) -> ModelBuilder:
    """对已是 (clin_dim,num_classes,df,clin_emb_dim,f_miss_dim) 形式的构造函数做简单包装。"""

    def _builder(clin_dim: int, num_classes: int, df: int, clin_emb_dim: int, f_miss_dim: int) -> nn.Module:
        return builder_fn(
            clin_dim=clin_dim,
            num_classes=num_classes,
            df=df,
            clin_emb_dim=clin_emb_dim,
            f_miss_dim=f_miss_dim,
        )

    return _builder


def _wrap_missing_aware_ablation(ablation: str | None, use_proco: bool = False) -> ModelBuilder:
    """构造 MissingAwareFusion / ProCo 版本的消融模型构造器。"""

    def _builder(clin_dim: int, num_classes: int, df: int, clin_emb_dim: int, f_miss_dim: int) -> nn.Module:
        if use_proco:
            return fusion.build_ablation_model_proco(
                ablation=ablation,
                clin_dim=clin_dim,
                num_classes=num_classes,
                df=df,
                clin_emb_dim=clin_emb_dim,
                f_miss_dim=f_miss_dim,
            )
        return fusion.build_ablation_model(
            ablation=ablation,
            clin_dim=clin_dim,
            num_classes=num_classes,
            df=df,
            clin_emb_dim=clin_emb_dim,
            f_miss_dim=f_miss_dim,
        )

    return _builder


def _wrap_daft_variant(name: str | None) -> ModelBuilder:
    """封装 DAFT / HyperFusion / DrFuse / Cross-Attn / Late 等变体。"""

    def _builder(clin_dim: int, num_classes: int, df: int, clin_emb_dim: int, f_miss_dim: int) -> nn.Module:
        return fusion.build_daft_or_baseline(
            name=name,
            clin_dim=clin_dim,
            num_classes=num_classes,
            df=df,
            clin_emb_dim=clin_emb_dim,
            f_miss_dim=f_miss_dim,
        )

    return _builder


MODEL_REGISTRY: Dict[str, ModelBuilder] = {
    # ----------------- 图像 only -----------------
    "image_resnet18": _wrap_image_only(backbones.ImageResNet18),
    "image_unet3d": _wrap_image_only(backbones.ImageUNet3D),
    "image_vit": _wrap_image_only(backbones.SimpleViT3D),

    # ----------------- 临床 only -----------------
    "clinical_mlp": _wrap_clinical_only(backbones.ClinicalMLP),
    "clinical_mlp_missing": _wrap_clinical_only(backbones.ClinicalMLPWithMissing),
    "clinical_ft_transformer": _wrap_clinical_only(backbones.FTTransformerClinical),

    # ----------------- 融合基线 -----------------
    "fusion_concat": _wrap_daft_variant("concat"),
    "fusion_cross_attn": _wrap_daft_variant("cross_attn"),
    "fusion_late": _wrap_daft_variant("late"),
    "fusion_daft": _wrap_daft_variant("daft"),
    "fusion_hyperfusion": _wrap_daft_variant("hyperfusion"),
    "fusion_drfuse": _wrap_daft_variant("drfuse"),

    # ----------------- Ours (Missing-aware Fusion) -----------------
    "ours": _wrap_missing_aware_ablation(ablation=None, use_proco=False),
    "ours_proco": _wrap_missing_aware_ablation(ablation=None, use_proco=True),
}


# 补充 Ours 消融（与原 ABLATION_NAMES 一致）
for abl in ["A1", "A2", "A3", "B1", "B2", "C1", "C2", "E1"]:
    MODEL_REGISTRY[f"ours_{abl}"] = _wrap_missing_aware_ablation(ablation=abl, use_proco=False)
    MODEL_REGISTRY[f"ours_proco_{abl}"] = _wrap_missing_aware_ablation(ablation=abl, use_proco=True)


def list_models() -> List[str]:
    """返回当前可选模型名称列表。"""
    return sorted(MODEL_REGISTRY.keys())


def build_model(
    name: str,
    clin_dim: int,
    num_classes: int,
    df: int = 256,
    clin_emb_dim: int = 256,
    f_miss_dim: int = 64,
) -> nn.Module:
    """
    构建指定名称的模型。

    Args:
        name: 模型名称（见 `list_models()`）
        clin_dim: 临床特征维度
        num_classes: 类别数
        df: 图像/融合特征维度（对多数模型为 256）
        clin_emb_dim: 临床 embedding 维度（对多数模型为 256）
        f_miss_dim: 缺失模式 embedding 维度
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"未知模型名称: {name}. 可选: {list_models()}")
    return MODEL_REGISTRY[name](clin_dim, num_classes, df, clin_emb_dim, f_miss_dim)


