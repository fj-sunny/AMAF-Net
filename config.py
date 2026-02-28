# -*- coding: utf-8 -*-
"""
全局配置与类别映射。数据路径不写死，请通过命令行或调用方传入。
"""
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# 数据与输出路径（占位，由调用方传入）
# ---------------------------------------------------------------------------
SRC_CT_ROOT = ""
SRC_CLINICAL_XLSX = ""
OUT_ROOT = "./output/missing_aware_fusion_5fold"

# ---------------------------------------------------------------------------
# 图像与任务
# ---------------------------------------------------------------------------
IMG_SIZE = (96, 96, 96)
USE_3_CLASS = False  # False = 4 类 PA/PC/CS/NFA

# ---------------------------------------------------------------------------
# 划分与随机种子
# ---------------------------------------------------------------------------
N_FOLDS = 5
TRAIN_TEST_RATIO = 0.8
TRAIN_VAL_RATIO = 0.8
SEED = 20260115

# ---------------------------------------------------------------------------
# 训练超参数
# ---------------------------------------------------------------------------
EPOCHS = 100
BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 0.1
NUM_WORKERS = 0
USE_EARLY_STOP = True
PATIENCE = 10
USE_LR_SCHEDULER = True
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_MIN = 1e-6
USE_CLASS_RESAMPLING = True
MINORITY_BOOST_FACTOR = 2.0
LAMBDA_PROTO = 0.0
MISSING_BUCKET_BOUNDS = (0.0, 0.2, 0.5, 1.01)
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = None


def get_class_maps(use_3_class: bool = False) -> Tuple[Dict[str, int], Dict[int, str], int]:
    """返回 (CLASS2IDX, IDX2CLASS, NUM_CLASSES)。"""
    if use_3_class:
        c2i = {"CS": 0, "PA": 1, "PC": 2}
    else:
        c2i = {"CS": 0, "PA": 1, "PC": 2, "NFA": 3}
    i2c = {v: k for k, v in c2i.items()}
    return c2i, i2c, len(c2i)


def get_minority_classes(num_classes: int) -> List[int]:
    return [0, 3] if num_classes == 4 else [0]


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
