"""
AMAF-Net（Adaptive Missing-Aware Fusion Network）核心代码包

本包在原有脚本 `train_missing_aware_fusion*.py` 基础上做了“规范化分模块”封装，
将数据处理、模型定义、训练与评估接口拆分为 models / utils / train / test 等子模块，
便于论文实验（基线 + 消融）统一管理与复现。

推荐在项目根目录运行，例如::

    python -m amaf_net.main --experiment ours_proco \
        --ct-root /path/to/your/ct_roi \
        --clinical-xlsx /path/to/your/clinical.xlsx \
        --out-root ./output/amaf_ours_proco

"""

__all__ = [
    "models",
    "utils",
]

