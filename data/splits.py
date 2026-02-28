# -*- coding: utf-8 -*-
"""训练/验证/测试划分与 5 折分层。"""
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold


def split_train_val_test_case_ids(
    case_ids: List[str],
    y_per_case: Dict[str, int],
    train_test_ratio: float = 0.8,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    ids = list(case_ids)
    y = np.array([y_per_case[cid] for cid in ids], dtype=int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(skf.split(ids, y))
    train_val_ids = [ids[i] for i in train_val_idx]
    test_ids = [ids[i] for i in test_idx]
    return train_val_ids, test_ids


def stratified_kfold_case_ids(
    case_ids: List[str], y_per_case: Dict[str, int], n_splits: int = 5, seed: int = 0
) -> List[List[str]]:
    ids = list(case_ids)
    y = np.array([y_per_case[cid] for cid in ids], dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for _, val_idx in skf.split(ids, y):
        folds.append([ids[i] for i in val_idx])
    return folds
