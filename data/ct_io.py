# -*- coding: utf-8 -*-
"""CT 路径解析、预处理与 NC 条目构建。"""
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np

from ..config import IMG_SIZE


_SUFFIX_PATTERN = re.compile(r"_(NC|PC|AP|VP|DP|EN|EX|PLAIN|PA|PV|\d+)$", re.IGNORECASE)


def normalize_id(x) -> str:
    if x is None:
        return ""
    s = str(x).strip().replace(" ", "")
    if s.endswith(".0"):
        s = s[:-2]
    s = _SUFFIX_PATTERN.sub("", s)
    return s


def parse_label_from_name(fname: str, class2idx: Dict[str, int]) -> int:
    prefix = fname.split("_", 1)[0]
    if prefix not in class2idx:
        raise ValueError(f"Unknown class prefix '{prefix}' in {fname}")
    return class2idx[prefix]


def parse_phase_from_name(fname: str) -> str:
    base = fname.replace(".nii.gz", "")
    parts = base.split("_")
    if len(parts) >= 3:
        pc = parts[-2].upper()
        if pc == "NC":
            return "NC"
        if pc in ["PA", "AP", "ARTERIAL"]:
            return "PA"
        if pc in ["PV", "VP", "VENOUS", "PORTAL"] or pc.isdigit():
            return "PV"
    for i in range(1, len(parts) - 1):
        pu = parts[i].upper()
        if pu == "NC":
            return "NC"
        if pu in ["PA", "AP", "ARTERIAL"]:
            return "PA"
        if pu in ["PV", "VP", "VENOUS", "PORTAL"] or pu.isdigit():
            return "PV"
    return "PV"


def parse_case_id(fname: str) -> str:
    base = fname.replace(".nii.gz", "")
    parts = base.split("_")
    if len(parts) >= 4:
        case_id_parts = parts[1:-2]
        if case_id_parts:
            return normalize_id("_".join(case_id_parts))
    if len(parts) >= 2:
        mid = "_".join(parts[1:])
        for phase in ["NC", "PA", "PV", "AP", "VP"]:
            if mid.endswith(f"_{phase}") or f"_{phase}_" in mid:
                mid = mid.rsplit(f"_{phase}", 1)[0]
                break
        return normalize_id(mid)
    return normalize_id(base)


def preprocess_ct(volume: np.ndarray, return_valid_mask: bool = False, img_size: Tuple[int, int, int] = IMG_SIZE):
    if volume.ndim == 4 and volume.shape[-1] == 1:
        volume = volume[..., 0]
    if tuple(volume.shape) != img_size:
        raise ValueError(f"ROI must be {img_size}, got {volume.shape}.")
    D, H, W = volume.shape
    valid_mask = np.ones((D, H, W), dtype=np.float32)
    d_nonzero = (volume > 1e-6).any(axis=(1, 2))
    d_indices = np.where(d_nonzero)[0]
    d_start, d_end = (d_indices[0], d_indices[-1] + 1) if len(d_indices) > 0 else (0, D)
    h_nonzero = (volume > 1e-6).any(axis=(0, 2))
    h_indices = np.where(h_nonzero)[0]
    h_start, h_end = (h_indices[0], h_indices[-1] + 1) if len(h_indices) > 0 else (0, H)
    w_nonzero = (volume > 1e-6).any(axis=(0, 1))
    w_indices = np.where(w_nonzero)[0]
    w_start, w_end = (w_indices[0], w_indices[-1] + 1) if len(w_indices) > 0 else (0, W)
    valid_mask[:d_start, :, :] = 0
    valid_mask[d_end:, :, :] = 0
    valid_mask[:, :h_start, :] = 0
    valid_mask[:, h_end:, :] = 0
    valid_mask[:, :, :w_start] = 0
    valid_mask[:, :, w_end:] = 0
    v = volume.astype(np.float32)
    m = valid_mask > 0.5
    if m.any():
        vv = v[m]
        mean, std = float(vv.mean()), float(vv.std()) + 1e-6
    else:
        mean, std = float(v.mean()), float(v.std()) + 1e-6
    v = (v - mean) / std
    v = v * valid_mask
    if return_valid_mask:
        return v, valid_mask
    return v


def scan_ct_items(
    src_root: str,
    use_3_class: bool,
    class2idx: Dict[str, int],
    idx2class: Dict[int, str],
) -> List[Dict[str, Any]]:
    src_root = Path(src_root)
    paths = sorted(src_root.rglob("*.nii.gz"))
    if not paths:
        raise FileNotFoundError(f"No .nii.gz in {src_root}")
    items = []
    for p in paths:
        fname = p.name
        if use_3_class and fname.split("_", 1)[0] == "NFA":
            continue
        try:
            y = parse_label_from_name(fname, class2idx)
            cid = parse_case_id(fname)
            phase = parse_phase_from_name(fname)
            items.append({
                "path": str(p), "fname": fname, "label": int(y),
                "class": idx2class[int(y)], "case_id": cid, "phase": phase,
            })
        except ValueError:
            continue
    return items


def group_by_case_and_phase(items: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    grouped = defaultdict(dict)
    for it in items:
        cid, phase = it["case_id"], it["phase"]
        if phase not in grouped[cid]:
            grouped[cid][phase] = it
    return dict(grouped)


def build_multiphase_items_all_data(
    grouped: Dict[str, Dict[str, Dict]],
    idx2class: Dict[int, str],
) -> List[Dict]:
    phases = ["NC", "PA", "PV"]
    out = []
    for cid, phases_dict in grouped.items():
        labels = [it["label"] for it in phases_dict.values()]
        label = labels[0]
        phase_paths = {}
        for phase in phases:
            phase_paths[phase] = phases_dict[phase]["path"] if phase in phases_dict else None
        out.append({
            "case_id": cid, "label": label, "class": idx2class[label],
            "phase_paths": phase_paths,
        })
    return out


def build_nc_items_from_multiphase(multiphase_items: List[Dict]) -> List[Dict]:
    out = []
    for it in multiphase_items:
        nc_path = it["phase_paths"].get("NC")
        if nc_path is None:
            raise ValueError(f"Case {it['case_id']} has no NC phase.")
        out.append({
            "case_id": it["case_id"], "label": int(it["label"]),
            "class": it["class"], "nc_path": nc_path,
        })
    return out
