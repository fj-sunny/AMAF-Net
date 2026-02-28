"""
指标计算封装

从原脚本导出常用指标计算函数，保证新的训练/评估脚本与旧实现保持一致：
- confusion_and_metrics: Acc / per-class Recall / Macro-F1 / Balanced Acc
- multiclass_ovr_auc: 多分类 one-vs-rest AUC
- compute_ece: 期望校准误差
- brier_score: Brier Score
- nll_score: 负对数似然
"""

from __future__ import annotations

from train_missing_aware_fusion import (  # type: ignore
    confusion_and_metrics,
    multiclass_ovr_auc,
    compute_ece,
    brier_score,
    nll_score,
)

__all__ = [
    "confusion_and_metrics",
    "multiclass_ovr_auc",
    "compute_ece",
    "brier_score",
    "nll_score",
]

