"""
五折训练。

ProCo 分支按 train_missing_aware_fusion_proco.py：
交叉熵（或 Focal）+ λ·ProCo，梯度裁剪，验证集 macro-F1 早停。
其余实验使用交叉熵，早停轮数用 config.PATIENCE。
数据来自本包的 data/，不再导入外部训练脚本。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

import config as cfg_mod
from models import registry as model_registry
from utils import data as data_utils
from utils import losses as loss_utils
from utils import metrics as metric_utils


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
    class2idx: Optional[Dict[str, int]] = None
    idx2class: Optional[Dict[int, str]] = None
    n_folds: int = 5


def _class_maps(cfg: ExperimentConfig):
    if cfg.class2idx is not None and cfg.idx2class is not None:
        return cfg.class2idx, cfg.idx2class
    class2idx, idx2class, _ = cfg_mod.get_class_maps(cfg.use_3_class)
    return class2idx, idx2class


def prepare_data_splits(cfg: ExperimentConfig):
    """加载 CT 与临床表，按病例做 8:2 再五折。"""
    class2idx, idx2class = _class_maps(cfg)
    all_ct_case_ids, matched_ct_items, all_multiphase = data_utils.load_ct_items(
        cfg.ct_root, use_3_class=cfg.use_3_class, class2idx=class2idx, idx2class=idx2class
    )
    x_df, y_clin, case_id_clin, class_names = data_utils.load_clinical_table(
        cfg.clinical_xlsx, use_3_class=cfg.use_3_class
    )
    ct_set = set(all_ct_case_ids)
    clin_set = set(case_id_clin)
    cases_with_nc = {it["case_id"] for it in all_multiphase if it["phase_paths"].get("NC")}
    matched_case_ids = sorted(ct_set & clin_set & cases_with_nc)
    if not matched_case_ids:
        raise ValueError("No matched cases between CT and clinical (with NC phase).")

    matched_ct_items = [it for it in matched_ct_items if it["case_id"] in matched_case_ids]
    clin_indices = [i for i, cid in enumerate(case_id_clin) if cid in set(matched_case_ids)]
    x_matched = x_df.iloc[clin_indices].reset_index(drop=True)
    y_matched = y_clin[clin_indices]
    case_id_matched = [case_id_clin[i] for i in clin_indices]
    y_per_case = dict(zip(case_id_matched, y_matched))

    train_val_ids, test_ids = data_utils.split_train_val_test_case_ids(
        matched_case_ids, y_per_case, train_test_ratio=cfg_mod.TRAIN_TEST_RATIO, seed=cfg_mod.SEED
    )
    folds_val_ids = data_utils.stratified_kfold_case_ids(
        train_val_ids, y_per_case, n_splits=cfg_mod.N_FOLDS, seed=cfg_mod.SEED
    )
    return (
        matched_ct_items,
        x_matched,
        y_matched,
        case_id_matched,
        train_val_ids,
        test_ids,
        folds_val_ids,
        class_names,
    )


def _build_weighted_sampler(items, num_classes: int, minority_classes: List[int], boost_factor: float):
    counts = np.zeros(num_classes, dtype=np.int64)
    for it in items:
        counts[int(it["label"])] += 1
    counts = np.maximum(counts, 1)
    class_weight = 1.0 / np.sqrt(counts.astype(np.float64))
    for c in minority_classes:
        if c < num_classes:
            class_weight[c] *= boost_factor
    sample_weights = [class_weight[int(it["label"])] for it in items]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def _build_dataloaders_for_fold(
    fold: int,
    matched_ct_items,
    x_matched,
    case_id_matched,
    train_val_ids,
    test_ids,
    folds_val_ids,
    use_missing_augmentation: bool,
    use_class_balance: bool,
    num_classes: int,
):
    val_case_ids = set(folds_val_ids[fold])
    train_case_ids = [c for c in train_val_ids if c not in val_case_ids]
    train_case_set, val_case_set, test_case_set = set(train_case_ids), val_case_ids, set(test_ids)

    train_ct_items = [it for it in matched_ct_items if it["case_id"] in train_case_set]
    val_ct_items = [it for it in matched_ct_items if it["case_id"] in val_case_set]
    test_ct_items = [it for it in matched_ct_items if it["case_id"] in test_case_set]

    train_clin_idx = [i for i, cid in enumerate(case_id_matched) if cid in train_case_set]
    val_clin_idx = [i for i, cid in enumerate(case_id_matched) if cid in val_case_set]
    test_clin_idx = [i for i, cid in enumerate(case_id_matched) if cid in test_case_set]

    x_train = x_matched.iloc[train_clin_idx].reset_index(drop=True)
    x_val = x_matched.iloc[val_clin_idx].reset_index(drop=True)
    x_test = x_matched.iloc[test_clin_idx].reset_index(drop=True)
    (
        xtr, xva, xte,
        m_train, m_val, m_test,
        r_train, r_val, r_test,
        _imputer, _scaler,
    ) = data_utils.clinical_fit_transform_fold(x_train, x_val, x_test)

    clin_dim = xtr.shape[1]
    train_clin_dict = {case_id_matched[train_clin_idx[i]]: i for i in range(len(train_clin_idx))}
    val_clin_dict = {case_id_matched[val_clin_idx[i]]: i for i in range(len(val_clin_idx))}
    test_clin_dict = {case_id_matched[test_clin_idx[i]]: i for i in range(len(test_clin_idx))}

    train_ds = data_utils.NCCTClinicalDataset(
        train_ct_items, train_clin_dict, xtr, m_train, r_train,
        use_missing_augmentation=use_missing_augmentation, missing_aug_prob=0.2,
    )
    val_ds = data_utils.NCCTClinicalDataset(val_ct_items, val_clin_dict, xva, m_val, r_val)
    test_ds = data_utils.NCCTClinicalDataset(test_ct_items, test_clin_dict, xte, m_test, r_test)

    if cfg_mod.USE_CLASS_RESAMPLING and use_class_balance:
        minority = cfg_mod.get_minority_classes(num_classes)
        sampler = _build_weighted_sampler(
            train_ct_items, num_classes, minority, cfg_mod.MINORITY_BOOST_FACTOR
        )
        train_loader = DataLoader(
            train_ds, batch_size=cfg_mod.BATCH_SIZE, sampler=sampler,
            shuffle=False, num_workers=cfg_mod.NUM_WORKERS,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg_mod.BATCH_SIZE, shuffle=True, num_workers=cfg_mod.NUM_WORKERS
        )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg_mod.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=cfg_mod.NUM_WORKERS)
    return train_loader, val_loader, test_loader, clin_dim, list(x_train.columns)


def _batch_to_device(batch, device: torch.device):
    x_ct, clin_v, clin_m, r_g, y, cids = batch
    x_ct = x_ct.to(device, non_blocking=True)
    clin_v = clin_v.to(device, non_blocking=True)
    clin_m = clin_m.to(device, non_blocking=True)
    r_g = r_g.to(device, non_blocking=True)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.long, device=device)
    else:
        y = y.to(device, non_blocking=True)
    return x_ct, clin_v, clin_m, r_g, y, cids


def fusion_embedding(aux: dict) -> torch.Tensor:
    """取出 ProCo 使用的融合特征。模型若已返回 f_fusion 则直接用。"""
    fused = aux.get("f_fusion")
    if fused is not None:
        return fused
    f_img = aux["f_img"]
    f_clin = aux["f_clin"]
    f_int = aux.get("f_int")
    alpha_img = aux["alpha_img"]
    alpha_clin = aux["alpha_clin"]
    alpha_int = aux.get("alpha_int")
    if f_int is None or alpha_int is None:
        return alpha_img * f_img + alpha_clin * f_clin
    return alpha_img * f_img + alpha_clin * f_clin + alpha_int * f_int


def _autocast(device: torch.device):
    if hasattr(torch, "amp"):
        return torch.amp.autocast("cuda", enabled=(device.type == "cuda"))
    return torch.cuda.amp.autocast(enabled=(device.type == "cuda"))


def _make_scaler(device: torch.device):
    enabled = device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _empty_metrics(num_classes: int) -> dict:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    prob = np.zeros((0, num_classes), dtype=np.float32)
    true = np.zeros((0,), dtype=np.int64)
    out = metric_utils.summarize_batch_metrics(cm, true, prob if prob.size else np.zeros((0, num_classes)))
    out["loss"] = 0.0
    return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ce_loss: nn.Module,
    num_classes: int,
    proco_loss: Optional[nn.Module] = None,
    lambda_proco: float = 0.0,
    scaler=None,
    clip_grad: Optional[float] = None,
):
    model.train()
    if proco_loss is not None:
        proco_loss.train()
    total_loss = total_ce = total_proco = 0.0
    total = 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    all_prob, all_true = [], []

    for batch in loader:
        x_ct, clin_v, clin_m, r_g, y, _ = _batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            logits, aux = model(x_ct, clin_v, clin_m, r_g)
            ce_val = ce_loss(logits, y)
            if lambda_proco > 0 and proco_loss is not None:
                fused = F.normalize(fusion_embedding(aux), dim=1)
                if fused.size(1) != proco_loss.feat_dim:
                    raise RuntimeError(
                        f"ProCo 特征维 {proco_loss.feat_dim} 与融合特征 {fused.size(1)} 不一致"
                    )
                proco_val = proco_loss(fused, y, update_prototypes=True)
                loss = ce_val + lambda_proco * proco_val
            else:
                proco_val = torch.zeros((), device=device)
                loss = ce_val
        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            if clip_grad is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], clip_grad)
                if proco_loss is not None and len(optimizer.param_groups) > 1:
                    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[1]["params"], clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                if proco_loss is not None:
                    torch.nn.utils.clip_grad_norm_(proco_loss.parameters(), clip_grad)
            optimizer.step()

        n = x_ct.size(0)
        total_loss += float(loss.detach()) * n
        total_ce += float(ce_val.detach()) * n
        total_proco += float(proco_val.detach()) * n
        total += n
        prob = F.softmax(logits.detach(), dim=1).cpu().numpy()
        pred = prob.argmax(axis=1)
        all_prob.append(prob)
        all_true.append(y.detach().cpu().numpy())
        for gt, pr in zip(y.detach().cpu().numpy(), pred):
            cm[int(gt), int(pr)] += 1

    if total == 0:
        out = _empty_metrics(num_classes)
        out.update({"ce_loss": 0.0, "proco_loss": 0.0})
        return out
    y_true = np.concatenate(all_true, axis=0)
    y_prob = np.concatenate(all_prob, axis=0)
    out = metric_utils.summarize_batch_metrics(cm, y_true, y_prob)
    out["loss"] = total_loss / total
    out["ce_loss"] = total_ce / total
    out["proco_loss"] = total_proco / total
    return out


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ce_loss: nn.Module,
    num_classes: int,
    return_details: bool = False,
) -> dict:
    model.eval()
    total_loss, total = 0.0, 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    all_prob, all_true = [], []
    all_case_ids, all_r_global = [], []
    all_alpha_img, all_alpha_clin, all_alpha_int = [], [], []

    for batch in loader:
        x_ct, clin_v, clin_m, r_g, y, cids = _batch_to_device(batch, device)
        logits, aux = model(x_ct, clin_v, clin_m, r_g)
        loss = ce_loss(logits, y)
        n = x_ct.size(0)
        total_loss += float(loss) * n
        total += n
        prob = F.softmax(logits, dim=1).cpu().numpy()
        pred = prob.argmax(axis=1)
        all_prob.append(prob)
        all_true.append(y.cpu().numpy())
        for gt, pr in zip(y.cpu().numpy(), pred):
            cm[int(gt), int(pr)] += 1
        if return_details:
            if isinstance(cids, (list, tuple)):
                all_case_ids.extend(list(cids))
            else:
                all_case_ids.append(cids)
            all_r_global.append(r_g.detach().cpu().numpy())
            all_alpha_img.append(aux["alpha_img"].detach().cpu().numpy().reshape(-1))
            all_alpha_clin.append(aux["alpha_clin"].detach().cpu().numpy().reshape(-1))
            all_alpha_int.append(aux["alpha_int"].detach().cpu().numpy().reshape(-1))

    if total == 0:
        return _empty_metrics(num_classes)
    y_true = np.concatenate(all_true, axis=0)
    y_prob = np.concatenate(all_prob, axis=0)
    result = metric_utils.summarize_batch_metrics(cm, y_true, y_prob)
    result["loss"] = total_loss / total
    if return_details:
        result["all_case_ids"] = all_case_ids
        result["all_pred"] = y_prob.argmax(axis=1)
        result["all_r_global"] = np.concatenate(all_r_global, axis=0) if all_r_global else np.array([])
        result["all_alpha_img"] = np.concatenate(all_alpha_img, axis=0) if all_alpha_img else np.array([])
        result["all_alpha_clin"] = np.concatenate(all_alpha_clin, axis=0) if all_alpha_clin else np.array([])
        result["all_alpha_int"] = np.concatenate(all_alpha_int, axis=0) if all_alpha_int else np.array([])
    return result


def _focal_from_loader(loader: DataLoader, num_classes: int, device: torch.device) -> nn.Module:
    labels = []
    for batch in loader:
        y = batch[4]
        if isinstance(y, torch.Tensor):
            labels.extend(y.cpu().numpy().tolist())
        else:
            labels.extend(list(y))
    counts = np.bincount(np.array(labels, dtype=int), minlength=num_classes).astype(np.float64)
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.sum() * num_classes
    return loss_utils.FocalLoss(alpha=weights.tolist(), gamma=2.0, num_classes=num_classes).to(device)


def _ensure_proco(proco_loss, fused, num_classes, temperature, lambda_uncertainty, device, optimizer):
    if proco_loss is not None:
        return proco_loss
    module = loss_utils.ProbabilisticContrastiveLoss(
        feat_dim=int(fused.size(1)),
        num_classes=num_classes,
        temperature=temperature,
        lambda_uncertainty=lambda_uncertainty,
        momentum=0.9,
    ).to(device)
    optimizer.add_param_group({
        "params": list(module.parameters()),
        "lr": optimizer.param_groups[0]["lr"],
        "weight_decay": optimizer.param_groups[0]["weight_decay"],
    })
    module.reset_prototypes()
    return module


def _save_fold_artifacts(test_m: dict, out_root: Path, fold: int, num_classes: int, idx2class: Dict[int, str]):
    out_root.mkdir(parents=True, exist_ok=True)
    class_names = [idx2class[c] for c in range(num_classes)]
    prob_cols = [f"prob_{idx2class[c]}" for c in range(num_classes)]
    rows = []
    r_global = test_m.get("all_r_global", np.array([]))
    for i, cid in enumerate(test_m.get("all_case_ids", [])):
        row = {
            "case_id": cid,
            "y_true": int(test_m["all_true"][i]),
            "y_pred": int(test_m["all_pred"][i]),
            "r_global": float(r_global[i]) if r_global.size else 0.0,
        }
        for c in range(num_classes):
            row[prob_cols[c]] = float(test_m["all_prob"][i, c])
        rows.append(row)
    if rows:
        path = out_root / f"pred_fold{fold}.csv"
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[Saved] {path}")

    a_img = test_m.get("all_alpha_img", np.array([]))
    a_clin = test_m.get("all_alpha_clin", np.array([]))
    a_int = test_m.get("all_alpha_int", np.array([]))
    y_true = test_m["all_true"]
    if a_img.size and a_clin.size and a_int.size:
        dep_overall = {
            "alpha_img": float(np.mean(a_img)),
            "alpha_clin": float(np.mean(a_clin)),
            "alpha_int": float(np.mean(a_int)),
            "image_ratio": float(np.mean(a_img + a_int)),
            "table_ratio": float(np.mean(a_clin)),
        }
        dep_per_class = []
        for c in range(num_classes):
            mask = y_true == c
            if mask.sum() == 0:
                dep_per_class.append({
                    "class": idx2class[c], "class_idx": c, "n": 0,
                    "alpha_img": None, "alpha_clin": None, "alpha_int": None,
                    "image_ratio": None, "table_ratio": None,
                })
            else:
                dep_per_class.append({
                    "class": idx2class[c], "class_idx": c, "n": int(mask.sum()),
                    "alpha_img": float(np.mean(a_img[mask])),
                    "alpha_clin": float(np.mean(a_clin[mask])),
                    "alpha_int": float(np.mean(a_int[mask])),
                    "image_ratio": float(np.mean((a_img + a_int)[mask])),
                    "table_ratio": float(np.mean(a_clin[mask])),
                })
        test_m["modality_dep_overall"] = dep_overall
        test_m["modality_dep_per_class"] = dep_per_class
        with open(out_root / f"modality_dependency_fold{fold}.json", "w", encoding="utf-8") as f:
            json.dump({"fold": fold, "overall": dep_overall, "per_class": dep_per_class}, f, indent=2, ensure_ascii=False)
    else:
        test_m["modality_dep_overall"] = None
        test_m["modality_dep_per_class"] = None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import precision_recall_curve, roc_curve

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(test_m["cm"], annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")
        ax.set_title(f"Fold {fold}")
        fig.tight_layout()
        fig.savefig(out_root / f"confusion_matrix_fold{fold}.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        for c in range(num_classes):
            fpr, tpr, _ = roc_curve((y_true == c).astype(int), test_m["all_prob"][:, c])
            ax.plot(fpr, tpr, label=class_names[c])
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.legend()
        ax.set_title(f"ROC fold {fold}")
        fig.tight_layout()
        fig.savefig(out_root / f"roc_fold{fold}.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        for c in range(num_classes):
            prec, rec, _ = precision_recall_curve((y_true == c).astype(int), test_m["all_prob"][:, c])
            ax.plot(rec, prec, label=class_names[c])
        ax.legend()
        ax.set_title(f"PR fold {fold}")
        fig.tight_layout()
        fig.savefig(out_root / f"pr_fold{fold}.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] 跳过第 {fold} 折图：{exc}")


def train_fold(
    fold: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    out_root: Path,
    num_classes: int,
    idx2class: Dict[int, str],
    lambda_proco: float = 0.0,
    proco_temperature: float = 0.07,
    proco_lambda_uncertainty: float = 0.1,
    use_focal_loss: bool = False,
    ablation_name: Optional[str] = None,
    patience: int = 10,
    clip_grad: Optional[float] = None,
) -> dict:
    """训练一折并在测试集上评估。"""
    if ablation_name == "E1-D1":
        lambda_proco = 0.0
    if ablation_name == "E1-D3":
        use_focal_loss = True

    model = model.to(device)
    if use_focal_loss:
        ce_loss = _focal_from_loader(train_loader, num_classes, device)
    else:
        ce_loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_mod.LR, weight_decay=cfg_mod.WEIGHT_DECAY)
    scaler = _make_scaler(device)
    scheduler = None
    if cfg_mod.USE_LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=cfg_mod.LR_SCHEDULER_FACTOR,
            patience=cfg_mod.LR_SCHEDULER_PATIENCE, min_lr=cfg_mod.LR_SCHEDULER_MIN,
        )

    model_dir = out_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = model_dir / f"best_fold{fold}_macroF1.pth"
    best_val_f1 = -1.0
    bad_epochs = 0
    proco_loss = None

    for epoch in range(1, cfg_mod.EPOCHS + 1):
        if lambda_proco > 0:
            # 第一批样本才能知道融合特征维度，损失模块在 epoch 内按需创建。
            # 这里先跑一个前向来对齐维度，避免把创建逻辑塞进每个 batch。
            if proco_loss is None:
                model.eval()
                with torch.no_grad():
                    probe = next(iter(train_loader))
                    x_ct, clin_v, clin_m, r_g, _, _ = _batch_to_device(probe, device)
                    _, aux = model(x_ct, clin_v, clin_m, r_g)
                    fused = fusion_embedding(aux)
                proco_loss = _ensure_proco(
                    None, fused, num_classes, proco_temperature, proco_lambda_uncertainty, device, optimizer
                )
            if epoch == 1:
                proco_loss.reset_prototypes()

        tr_m = train_one_epoch(
            model, train_loader, optimizer, device, ce_loss, num_classes,
            proco_loss=proco_loss, lambda_proco=lambda_proco, scaler=scaler, clip_grad=clip_grad,
        )
        if proco_loss is not None:
            proco_loss.sync_prototypes()
        va_m = eval_one_epoch(model, val_loader, device, ce_loss, num_classes, return_details=False)
        if scheduler is not None:
            scheduler.step(va_m["macro_f1"])
        current_lr = optimizer.param_groups[0]["lr"]
        loss_info = f"λ_ProCo={lambda_proco:.2f}" if lambda_proco > 0 else "CE only"
        if use_focal_loss:
            loss_info += " (Focal)"
        print(f"[Fold {fold} Epoch {epoch:03d}] LR={current_lr:.2e} {loss_info}")
        proco_str = f", ProCo={tr_m.get('proco_loss', 0.0):.4f}" if lambda_proco > 0 else ""
        print(
            f"  Train loss={tr_m['loss']:.4f} (CE={tr_m.get('ce_loss', tr_m['loss']):.4f}{proco_str}) | "
            + metric_utils.format_metrics_line(tr_m, num_classes, idx2class)
        )
        print(
            f"  Val   loss={va_m['loss']:.4f} | "
            + metric_utils.format_metrics_line(va_m, num_classes, idx2class)
        )
        if va_m["macro_f1"] > best_val_f1:
            best_val_f1 = va_m["macro_f1"]
            save_dict = {"model_state_dict": model.state_dict()}
            if proco_loss is not None:
                save_dict["proco_loss_state_dict"] = proco_loss.state_dict()
            torch.save(save_dict, str(best_ckpt))
            bad_epochs = 0
        else:
            bad_epochs += 1
        if cfg_mod.USE_EARLY_STOP and bad_epochs >= patience:
            print(f"[Fold {fold}] Early stopping. Best val macroF1={best_val_f1:.4f}")
            break

    if best_ckpt.exists():
        checkpoint = torch.load(str(best_ckpt), map_location="cpu")
        state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state)
        if proco_loss is not None and isinstance(checkpoint, dict) and "proco_loss_state_dict" in checkpoint:
            proco_loss.load_state_dict(checkpoint["proco_loss_state_dict"])
        model.to(device)
        if proco_loss is not None:
            proco_loss.to(device)

    test_m = eval_one_epoch(model, test_loader, device, ce_loss, num_classes, return_details=True)
    pre_per, macro_pre, spe_per, macro_spe = metric_utils.precision_specificity_from_cm(test_m["cm"], num_classes)
    test_m["precision_per_class"] = pre_per
    test_m["macro_precision"] = macro_pre
    test_m["specificity_per_class"] = spe_per
    test_m["macro_specificity"] = macro_spe
    if len(test_m["all_true"]):
        test_m["nll"] = metric_utils.nll_score(test_m["all_true"], test_m["all_prob"])
        test_m["brier"] = metric_utils.brier_score(test_m["all_true"], test_m["all_prob"], num_classes)
        test_m["ece"] = metric_utils.compute_ece(test_m["all_true"], test_m["all_prob"])
        test_m["weighted_f1"] = metric_utils.weighted_f1(test_m["all_true"], test_m["all_pred"])
        test_m["auprc_macro"] = metric_utils.macro_average_precision_score(
            test_m["all_true"], test_m["all_prob"], num_classes
        )
    print(f"[Fold {fold} TEST] " + metric_utils.format_metrics_line(test_m, num_classes, idx2class))
    _save_fold_artifacts(test_m, out_root, fold, num_classes, idx2class)
    return test_m


def run_experiment(cfg: ExperimentConfig, num_classes: int, idx2class: dict):
    """按 experiment 名称构建模型并完成五折训练。"""
    cfg_mod.set_seed(cfg_mod.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    cfg.idx2class = idx2class

    (
        matched_ct_items,
        x_matched,
        _y_matched,
        case_id_matched,
        train_val_ids,
        test_ids,
        folds_val_ids,
        _class_names,
    ) = prepare_data_splits(cfg)

    fold_metrics = []
    patience = cfg_mod.PROCO_PATIENCE if cfg.use_proco else cfg_mod.PATIENCE
    clip_grad = cfg_mod.CLIP_GRAD_NORM if cfg.use_proco else None
    lambda_proco = cfg.lambda_proco if cfg.use_proco else 0.0
    n_folds = max(1, min(int(cfg.n_folds), len(folds_val_ids)))

    for fold in range(n_folds):
        train_loader, val_loader, test_loader, clin_dim, _feature_names = _build_dataloaders_for_fold(
            fold,
            matched_ct_items,
            x_matched,
            case_id_matched,
            train_val_ids,
            test_ids,
            folds_val_ids,
            use_missing_augmentation=cfg.use_missing_augmentation,
            use_class_balance=cfg.use_class_balance and not cfg.use_proco,
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
        test_m = train_fold(
            fold=fold,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model=model,
            device=device,
            out_root=cfg.out_root,
            num_classes=num_classes,
            idx2class=idx2class,
            lambda_proco=lambda_proco,
            proco_temperature=cfg.proco_temperature,
            proco_lambda_uncertainty=cfg.proco_lambda_uncertainty,
            use_focal_loss=False,
            ablation_name=cfg.ablation_name,
            patience=patience,
            clip_grad=clip_grad,
        )
        fold_metrics.append(test_m)
    return fold_metrics
