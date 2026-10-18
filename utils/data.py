"""
CT 与临床数据加载。

实现位于 data/ 包。本模块只按训练入口需要的形状做一次封装。
"""

from __future__ import annotations

from typing import Dict, List

from config import IMG_SIZE, MISSING_BUCKET_BOUNDS
from data import (
    NCCTClinicalDataset,
    build_multiphase_items_all_data,
    build_nc_items_from_multiphase,
    clinical_fit_transform_fold,
    group_by_case_and_phase,
    load_clinical_xlsx,
    scan_ct_items,
    split_train_val_test_case_ids as _split_train_val_test_case_ids,
    stratified_kfold_case_ids as _stratified_kfold_case_ids,
)

load_clinical_xlsx = load_clinical_xlsx


def load_ct_items(ct_root: str, use_3_class: bool, class2idx: Dict[str, int], idx2class: Dict[int, str]):
    """扫描 CT ROI，返回病例 ID、NC 条目、以及全部相位条目。"""
    ct_items_raw = scan_ct_items(ct_root, use_3_class=use_3_class, class2idx=class2idx, idx2class=idx2class)
    grouped = group_by_case_and_phase(ct_items_raw)
    all_multiphase = build_multiphase_items_all_data(grouped, idx2class)
    all_ct_case_ids = [it["case_id"] for it in all_multiphase]
    matched_ct_items = build_nc_items_from_multiphase(
        [it for it in all_multiphase if it["phase_paths"].get("NC")]
    )
    return all_ct_case_ids, matched_ct_items, all_multiphase


def load_clinical_table(clinical_xlsx: str, use_3_class: bool):
    """读取临床 Excel，返回 X_df, y, case_id, class_names。"""
    return load_clinical_xlsx(clinical_xlsx, use_3_class=use_3_class)


def split_train_val_test_case_ids(case_ids: List[str], y_per_case: Dict[str, int], train_test_ratio: float, seed: int):
    return _split_train_val_test_case_ids(
        case_ids=case_ids,
        y_per_case=y_per_case,
        train_test_ratio=train_test_ratio,
        seed=seed,
    )


def stratified_kfold_case_ids(case_ids: List[str], y_per_case: Dict[str, int], n_splits: int, seed: int):
    return _stratified_kfold_case_ids(case_ids, y_per_case, n_splits=n_splits, seed=seed)


__all__ = [
    "IMG_SIZE",
    "MISSING_BUCKET_BOUNDS",
    "NCCTClinicalDataset",
    "clinical_fit_transform_fold",
    "load_ct_items",
    "load_clinical_table",
    "load_clinical_xlsx",
    "split_train_val_test_case_ids",
    "stratified_kfold_case_ids",
]
