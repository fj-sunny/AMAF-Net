"""
损失函数封装

统一导出：
- 交叉熵（直接使用 torch.nn.CrossEntropyLoss）
- FocalLoss（train_missing_aware_fusion 与 *_proco 中的两个版本）
- ProbabilisticContrastiveLoss（ProCo，用于 Ours+ProCo）
"""

from __future__ import annotations

import torch.nn as nn

from train_missing_aware_fusion import FocalLoss as _FocalLoss_base  # type: ignore
from train_missing_aware_fusion_proco import (  # type: ignore
    FocalLoss as _FocalLoss_proco,
    ProbabilisticContrastiveLoss,
)


CrossEntropyLoss = nn.CrossEntropyLoss
FocalLossBase = _FocalLoss_base
FocalLossProCo = _FocalLoss_proco

__all__ = [
    "CrossEntropyLoss",
    "FocalLossBase",
    "FocalLossProCo",
    "ProbabilisticContrastiveLoss",
]

