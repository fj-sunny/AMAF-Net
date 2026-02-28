"""
amaf_net.models

模型子模块：
- backbones: 影像与临床 backbone（ResNet18-3D、3D U-Net、ViT、MLP、FT-Transformer 等）
- fusion: 各类多模态融合模型（Missing-aware Fusion / DAFT / Concat / Cross-Attn / Late / HyperFusion / DrFuse 等）
- registry: 统一的模型注册与选择接口
"""

from . import backbones  # noqa: F401
from . import fusion     # noqa: F401
from .registry import build_model, list_models  # noqa: F401

__all__ = ["backbones", "fusion", "build_model", "list_models"]

