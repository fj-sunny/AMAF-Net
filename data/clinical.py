# -*- coding: utf-8 -*-
"""临床表格加载与每折拟合变换。"""
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .ct_io import normalize_id


def load_clinical_xlsx(path: str, use_3_class: bool = False):
    df = pd.read_excel(path)
    if "tumor_type" not in df.columns:
        raise ValueError("Missing tumor_type")
    if "matched_folder" in df.columns:
        df["case_id"] = df["matched_folder"].astype(str)
    elif "case_id" not in df.columns:
        df["case_id"] = np.arange(len(df)).astype(str)
    df["case_id"] = df["case_id"].astype(str).apply(normalize_id)
    if use_3_class:
        keep = ["CS", "PA", "PC"]
        df = df[df["tumor_type"].astype(str).isin(keep)].copy()
        y_raw = df["tumor_type"].astype(str).values
        class_to_idx = {c: i for i, c in enumerate(keep)}
    else:
        y_raw = df["tumor_type"].astype(str).values
        lock = ["CS", "NFA", "PA", "PC"]
        class_names = lock if set(np.unique(y_raw)).issubset(set(lock)) else sorted(np.unique(y_raw).tolist())
        class_to_idx = {c: i for i, c in enumerate(class_names)}
    y = np.array([class_to_idx[str(v)] for v in y_raw], dtype=int)
    drop_cols = [c for c in ["tumor_type", "matched_folder"] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore").copy()
    case_id = X["case_id"].values
    X = X.drop(columns=["case_id"], errors="ignore")
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, y, case_id, list(class_to_idx.keys())


def clinical_fit_transform_fold(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, object]:
    M_train = X_train.isna().astype(np.float32).values
    M_val = X_val.isna().astype(np.float32).values
    M_test = X_test.isna().astype(np.float32).values
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xtr_imp = imputer.fit_transform(X_train.values).astype(np.float32)
    Xva_imp = imputer.transform(X_val.values).astype(np.float32)
    Xte_imp = imputer.transform(X_test.values).astype(np.float32)
    Xtr = scaler.fit_transform(Xtr_imp).astype(np.float32)
    Xva = scaler.transform(Xva_imp).astype(np.float32)
    Xte = scaler.transform(Xte_imp).astype(np.float32)
    r_train_global = M_train.mean(axis=1).astype(np.float32)
    r_val_global = M_val.mean(axis=1).astype(np.float32)
    r_test_global = M_test.mean(axis=1).astype(np.float32)
    return Xtr, Xva, Xte, M_train, M_val, M_test, r_train_global, r_val_global, r_test_global, imputer, scaler
