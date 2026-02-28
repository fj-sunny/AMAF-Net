# -*- coding: utf-8 -*-
"""NC CT + 临床 (C_tilde, R, r_global) Dataset。"""
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib

from .ct_io import preprocess_ct
from ..config import IMG_SIZE


class NCCTClinicalDataset(Dataset):
    def __init__(
        self,
        ct_items: List[dict],
        clinical_dict: Dict[str, int],
        clinical_v: np.ndarray,
        clinical_m: np.ndarray,
        r_global: np.ndarray,
        use_missing_augmentation: bool = False,
        missing_aug_prob: float = 0.2,
        img_size: tuple = IMG_SIZE,
    ):
        self.ct_items = ct_items
        self.clinical_dict = clinical_dict
        self.clinical_v = torch.tensor(clinical_v, dtype=torch.float32)
        self.clinical_m = torch.tensor(clinical_m, dtype=torch.float32)
        self.r_global = torch.tensor(r_global, dtype=torch.float32)
        self.use_missing_augmentation = use_missing_augmentation
        self.missing_aug_prob = missing_aug_prob
        self.img_size = img_size

    def __len__(self):
        return len(self.ct_items)

    def __getitem__(self, idx):
        it = self.ct_items[idx]
        cid, y = it["case_id"], int(it["label"])
        path = it.get("nc_path")
        if not path:
            raise ValueError(f"Missing nc_path for {cid}")
        img = nib.load(path)
        vol = np.asarray(img.get_fdata())
        vol, _ = preprocess_ct(vol, return_valid_mask=True, img_size=self.img_size)
        x_ct = torch.from_numpy(vol[None, ...].astype(np.float32))
        if cid in self.clinical_dict:
            i = self.clinical_dict[cid]
            clin_v = self.clinical_v[i].clone()
            clin_m = self.clinical_m[i].clone()
            r_g = self.r_global[i].clone()
        else:
            clin_v = torch.zeros(self.clinical_v.shape[1], dtype=torch.float32)
            clin_m = torch.ones(self.clinical_m.shape[1], dtype=torch.float32)
            r_g = torch.tensor(1.0, dtype=torch.float32)
        if self.use_missing_augmentation and clin_v.numel() > 0:
            d = clin_v.numel()
            mask_extra = torch.rand(d, dtype=torch.float32) < self.missing_aug_prob
            clin_v[mask_extra] = 0.0
            clin_m[mask_extra] = 1.0
            r_g = clin_m.mean()
        return x_ct, clin_v, clin_m, r_g, y, cid
