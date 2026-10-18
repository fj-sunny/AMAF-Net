"""
AMAF-Net（Adaptive Missing-Aware Fusion Network）核心代码包

数据处理、模型、损失和五折训练都在本包内。
在项目根目录运行::

    python main.py --experiment ours_proco \
        --ct-root /path/to/your/ct_roi \
        --clinical-xlsx /path/to/your/clinical.xlsx \
        --out-root ./output/amaf_ours_proco

"""

__all__ = [
    "models",
    "utils",
]

