"""
数据加载与划分工具封装

为避免重复实现，这里直接从原始脚本导入核心数据处理逻辑，并提供更清晰的函数封装：

- load_ct_items
- load_clinical_table
- split_train_val_test_case_ids
- stratified_kfold_case_ids
- clinical_fit_transform_fold
- NCCTClinicalDataset

这样在新的训练脚本中只需依赖本模块，而不直接依赖庞大的 train_missing_aware_fusion.py。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from train_missing_aware_fusion import (  # type: ignore
    IMG_SIZE,
    MISSING_BUCKET_BOUNDS,
    NCCTClinicalDataset as _NCCTClinicalDataset,
    clinical_fit_transform_fold as _clinical_fit_transform_fold,
    load_clinical_xlsx as _load_clinical_xlsx,
    scan_ct_items as _scan_ct_items,
    group_by_case_and_phase as _group_by_case_and_phase,
    build_multiphase_items_all_data as _build_multiphase_items_all_data,
    build_nc_items_from_multiphase as _build_nc_items_from_multiphase,
    split_train_val_test_case_ids as _split_train_val_test_case_ids,
    stratified_kfold_case_ids as _stratified_kfold_case_ids,
)

NCCTClinicalDataset = _NCCTClinicalDataset
clinical_fit_transform_fold = _clinical_fit_transform_fold
load_clinical_xlsx = _load_clinical_xlsx


def load_ct_items(ct_root: str):
    """扫描 CT ROI，并构建按 case 与 phase 聚合后的 NC 相位条目。"""
    ct_items_raw = _scan_ct_items(ct_root)
    grouped = _group_by_case_and_phase(ct_items_raw)
    all_multiphase = _build_multiphase_items_all_data(grouped)
    all_ct_case_ids = [it["case_id"] for it in all_multiphase]
    matched_ct_items = _build_nc_items_from_multiphase(
        [it for it in all_multiphase if it["phase_paths"].get("NC")]
    )
    return all_ct_case_ids, matched_ct_items, all_multiphase


def load_clinical_table(clinical_xlsx: str, use_3_class: bool):
    """
    读取临床 Excel，返回:
        X_df, y_clin, case_id_clin, class_names
    与原函数完全一致，仅做薄封装。
    """
    return _load_clinical_xlsx(clinical_xlsx, use_3_class=use_3_class)


def split_train_val_test_case_ids(
    case_ids: List[str],
    y_per_case: Dict[str, int],
    train_test_ratio: float,
    seed: int,
):
    """封装原始脚本中的按病例分层 Train+Val / Test 划分函数。"""
    return _split_train_val_test_case_ids(
        case_ids=case_ids,
        y_per_case=y_per_case,
        train_test_ratio=train_test_ratio,
        seed=seed,
    )


def stratified_kfold_case_ids(
    case_ids: List[str], y_per_case: Dict[str, int], n_splits: int, seed: int
):
    """封装原始脚本中的 5-fold 分层划分函数。"""
    return _stratified_kfold_case_ids(case_ids, y_per_case, n_splits=n_splits, seed=seed)


