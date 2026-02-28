"""
统一训练入口（薄封装）

为尽量复用原始脚本中已经验证过的训练逻辑，这里不重新实现完整的 5-fold 流程，
而是提供一个统一的“模型构建 + 数据准备”入口，然后调用：
- `train_missing_aware_fusion.train_fold`
- 或 `train_missing_aware_fusion_proco.train_fold_proco`
等已有函数。

在上层 `amaf_net.main` 中会根据 experiment / model 名称选择合适的训练函数。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_missing_aware_fusion import (  # type: ignore
    BATCH_SIZE,
    NUM_WORKERS,
    N_FOLDS,
    TRAIN_TEST_RATIO,
    SEED,
    MINORITY_CLASSES,
    MINORITY_BOOST_FACTOR,
    USE_CLASS_RESAMPLING,
    set_seed,
    build_weighted_sampler_minority_boost,
    train_fold as base_train_fold,
)
from train_missing_aware_fusion_proco import (  # type: ignore
    train_fold_proco,
)

from ..models import registry as model_registry
from ..utils import data as data_utils


@dataclass
class ExperimentConfig:
    ct_root: str
    clinical_xlsx: str
    out_root: Path
    use_3_class: bool = False
    model_name: str = "ours"
    use_proco: bool = False
    ablation_name: Optional[str] = None
    lambda_proco: float = 0.1
    proco_temperature: float = 0.07
    proco_lambda_uncertainty: float = 0.1
    use_missing_augmentation: bool = True
    use_class_balance: bool = True


def prepare_data_splits(cfg: ExperimentConfig):
    """
    复用原脚本的数据加载与划分逻辑，返回：
        matched_ct_items, X_matched, y_matched, case_id_matched,
        train_val_ids, test_ids, folds_val_ids
    """
    # CT
    all_ct_case_ids, matched_ct_items, all_multiphase = data_utils.load_ct_items(cfg.ct_root)

    # 临床
    X_df, y_clin, case_id_clin, class_names = data_utils.load_clinical_table(
        cfg.clinical_xlsx, use_3_class=cfg.use_3_class
    )
    ct_set = set(all_ct_case_ids)
    clin_set = set(case_id_clin)
    cases_with_nc = {it["case_id"] for it in all_multiphase if it["phase_paths"].get("NC")}
    matched_case_ids = sorted(list(ct_set & clin_set & cases_with_nc))
    if not matched_case_ids:
        raise ValueError("No matched cases between CT and clinical (with NC phase).")

    matched_ct_items = [it for it in matched_ct_items if it["case_id"] in matched_case_ids]
    clin_indices = [i for i, cid in enumerate(case_id_clin) if cid in matched_case_ids]
    X_matched = X_df.iloc[clin_indices].reset_index(drop=True)
    y_matched = y_clin[clin_indices]
    case_id_matched = [case_id_clin[i] for i in clin_indices]
    y_per_case = dict(zip(case_id_matched, y_matched))

    train_val_ids, test_ids = data_utils.split_train_val_test_case_ids(
        matched_case_ids, y_per_case, train_test_ratio=TRAIN_TEST_RATIO, seed=SEED
    )
    folds_val_ids = data_utils.stratified_kfold_case_ids(
        train_val_ids, y_per_case, n_splits=N_FOLDS, seed=SEED
    )
    return (
        matched_ct_items,
        X_matched,
        y_matched,
        case_id_matched,
        train_val_ids,
        test_ids,
        folds_val_ids,
        class_names,
    )


def _build_dataloaders_for_fold(
    fold: int,
    matched_ct_items,
    X_matched,
    case_id_matched,
    train_val_ids,
    test_ids,
    folds_val_ids,
    use_missing_augmentation: bool,
    use_class_balance: bool,
    num_classes: int,
):
    """为单个 fold 构建 train/val/test DataLoader，与原脚本保持一致。"""
    val_case_ids = folds_val_ids[fold]
    train_case_ids = [c for c in train_val_ids if c not in val_case_ids]
    test_case_ids = test_ids

    train_ct_items = [it for it in matched_ct_items if it["case_id"] in train_case_ids]
    val_ct_items = [it for it in matched_ct_items if it["case_id"] in val_case_ids]
    test_ct_items = [it for it in matched_ct_items if it["case_id"] in test_case_ids]

    train_clin_idx = [i for i, cid in enumerate(case_id_matched) if cid in train_case_ids]
    val_clin_idx = [i for i, cid in enumerate(case_id_matched) if cid in val_case_ids]
    test_clin_idx = [i for i, cid in enumerate(case_id_matched) if cid in test_case_ids]

    X_train = X_matched.iloc[train_clin_idx].reset_index(drop=True)
    X_val = X_matched.iloc[val_clin_idx].reset_index(drop=True)
    X_test = X_matched.iloc[test_clin_idx].reset_index(drop=True)

    (
        Xtr,
        Xva,
        Xte,
        M_train,
        M_val,
        M_test,
        r_train_global,
        r_val_global,
        r_test_global,
        imputer,
        scaler,
    ) = data_utils.clinical_fit_transform_fold(X_train, X_val, X_test)

    clin_dim = Xtr.shape[1]
    train_clin_dict = {case_id_matched[train_clin_idx[i]]: i for i in range(len(train_clin_idx))}
    val_clin_dict = {case_id_matched[val_clin_idx[i]]: i for i in range(len(val_clin_idx))}
    test_clin_dict = {case_id_matched[test_clin_idx[i]]: i for i in range(len(test_clin_idx))}

    train_ds = data_utils.NCCTClinicalDataset(
        train_ct_items,
        train_clin_dict,
        Xtr,
        M_train,
        r_train_global,
        use_missing_augmentation=use_missing_augmentation,
        missing_aug_prob=0.2,
    )
    val_ds = data_utils.NCCTClinicalDataset(
        val_ct_items, val_clin_dict, Xva, M_val, r_val_global
    )
    test_ds = data_utils.NCCTClinicalDataset(
        test_ct_items, test_clin_dict, Xte, M_test, r_test_global
    )

    if USE_CLASS_RESAMPLING and use_class_balance:
        sampler, _ = build_weighted_sampler_minority_boost(
            train_ct_items,
            num_classes=num_classes,
            minority_classes=MINORITY_CLASSES if num_classes == 4 else [0],
            boost_factor=MINORITY_BOOST_FACTOR,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    return (
        train_loader,
        val_loader,
        test_loader,
        clin_dim,
        (X_train.columns.tolist()),
    )


def run_experiment(cfg: ExperimentConfig, num_classes: int, idx2class: dict):
    """
    统一的 5-fold 训练入口。

    - 根据 cfg.model_name 通过 registry 构建模型
    - 通过原脚本的数据管线构建 DataLoader
    - 对于 use_proco=True 的实验，调用 train_fold_proco
      否则调用 base_train_fold
    - 保留原脚本的早停与指标统计逻辑
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.out_root.mkdir(parents=True, exist_ok=True)

    (
        matched_ct_items,
        X_matched,
        y_matched,
        case_id_matched,
        train_val_ids,
        test_ids,
        folds_val_ids,
        class_names,
    ) = prepare_data_splits(cfg)

    class_names_list = [idx2class[i] for i in range(num_classes)]

    fold_metrics = []

    for fold in range(N_FOLDS):
        (
            train_loader,
            val_loader,
            test_loader,
            clin_dim,
            clin_feature_names,
        ) = _build_dataloaders_for_fold(
            fold,
            matched_ct_items,
            X_matched,
            case_id_matched,
            train_val_ids,
            test_ids,
            folds_val_ids,
            use_missing_augmentation=cfg.use_missing_augmentation,
            use_class_balance=cfg.use_class_balance,
            num_classes=num_classes,
        )

        model = model_registry.build_model(
            name=cfg.model_name,
            clin_dim=clin_dim,
            num_classes=num_classes,
            df=256,
            clin_emb_dim=256,
            f_miss_dim=64,
        )

        if cfg.use_proco:
            test_m = train_fold_proco(
                fold=fold,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                model=model,
                device=device,
                out_root=cfg.out_root,
                clin_dim=clin_dim,
                num_classes=num_classes,
                lambda_proco=cfg.lambda_proco,
                proco_temperature=cfg.proco_temperature,
                proco_lambda_uncertainty=cfg.proco_lambda_uncertainty,
                clin_feature_names=clin_feature_names,
                use_focal_loss=False,
                ablation_name=cfg.ablation_name,
            )
        else:
            test_m = base_train_fold(
                fold=fold,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                model=model,
                device=device,
                out_root=cfg.out_root,
                clin_dim=clin_dim,
                num_classes=num_classes,
                use_class_balance=cfg.use_class_balance,
                criterion=None,
            )

        fold_metrics.append(test_m)

    # 汇总逻辑保持简单：交由原脚本在各自 out_root 下的 summary 文件中呈现；
    # 此处仅返回每折结果，供上层 main 打印/保存。
    return fold_metrics


