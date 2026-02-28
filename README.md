# Missingness-aware Multimodal Learning for Four-class Adrenal Tumor Classification

![Framework](./assets/framework.svg)

**AMAF-Net** (Adaptive Missing-Aware Fusion Network) is a missingness-aware multimodal classification framework for **non-contrast CT (NCCT) + clinical tabular data**. This repository contains the full implementation. Run experiments from the **AMAF-Net** directory (project root).

---

## Quick Start

**List available models / experiments:**

```bash
python main.py --list-models
```

**Run 5-fold training (full AMAF-Net model):**

```bash
python main.py --experiment ours_proco \
  --ct-root /path/to/ct_roi \
  --clinical-xlsx /path/to/clinical.xlsx \
  --out-root ./output/amaf_net
```

**Arguments:**

- `--experiment`: Model/experiment name (see `--list-models`; e.g. `image_resnet18`, `fusion_concat`, `ours`, `ours_proco`, `ours_A1`). `ours_proco` is the full AMAF-Net with probabilistic contrastive regularization on fusion features.
- `--ct-root`: Path to NCCT ROI volumes (not included; prepare locally).
- `--clinical-xlsx`: Path to clinical Excel file; format should match **`Adrenal_clinical_EN.xlsx`** (see Dataset & Clinical Features below). Prepare locally.
- `--out-root`: Output directory for logs and results.

---

## Repository Structure (AMAF-Net)

| Path | Description |
|------|-------------|
| **`config.py`** | Global config (image size, batch size, folds, seed, etc.). |
| **`main.py`** | Entry point: model selection and 5-fold training. |
| **`data/`** | **`ct_io.py`**: CT ROI scanning, phase grouping, preprocessing (resize to fixed voxel size, e.g. 96³). **`clinical.py`**: Load clinical Excel, labels, missing mask, standardization. **`dataset.py`**: `NCCTClinicalDataset` returning \(x_{ct}, \tilde{C}, R, r_{global}\). **`splits.py`**: Train/val/test and stratified K-fold by case ID. |
| **`models/`** | **`backbones.py`**: Image-only (3D ResNet18, 3D U-Net, ViT) and clinical-only (MLP, MLP+Missing, FT-Transformer) baselines. **`resnet3d.py`**, **`clinical_encoders.py`**, **`fusion_weights.py`**: 3D ResNet18, clinical encoders, missing-pattern MLP, fusion weight generator. **`missing_aware_fusion.py`**: AMAF backbone (Ours) and ablations (A1–E1) with FiLM and adaptive weights. **`fusion.py`**: Ours / DAFT / HyperFusion / Late Fusion / DrFuse. **`registry.py`**: Model registry (name → architecture). |
| **`train/trainer.py`** | 5-fold training: data splits, DataLoaders, training loop (including optional probabilistic contrastive regularization). |
| **`utils/`** | Data helpers, metrics, losses, visualization. |

Pipeline: **Load data → Build \((x_{ct}, \tilde{C}, R, r_{global})\) → Choose `--experiment` → 5-fold train & evaluate (AUC, Macro-F1, Acc, BAcc, etc.).**

---

## Dataset & Clinical Features

### Imaging (NCCT)

- Input: **NC-phase ROI volumes** (e.g. adrenal tumor) from single- or multi-phase CT. **`data/ct_io.py`** resamples to a fixed size (e.g. \(96\times96\times96\)).
- The code only uses paths and voxel arrays; no raw DICOM or patient metadata.

### Clinical table (Excel)

The experiments use a table in the format of **`Adrenal_clinical_EN.xlsx`** (column names can be mapped from a Chinese version via a script like `english_clinical.py`).

**Required columns (not used as features):**

- **`tumor_type`**: Label (e.g. `CS`, `PA`, `PC`, `NFA`) for 3- or 4-class classification.
- **`case_id`** or **`matched_folder`**: Case identifier that matches the CT ROI directory names (one-to-one with imaging).

**Feature columns (all other numeric columns):**  
Any remaining columns are treated as numeric clinical features (NaN = missing). In **`Adrenal_clinical_EN.xlsx`** these include (aligned with the English column mapping):

- **Demographics:** `sex` (0/1), `age`
- **Vitals:** `pulse`, `sbp`, `dbp`
- **History (0/1):** `hypertension`, `diabetes`, `abnormal_glucose`
- **Tumor / imaging-derived:** `dmax_mm`, `dmin_mm`, `aspect`, `log_volume`, `multi_lesion`
- **Serum electrolytes:** `potassium`, `sodium`, `chloride`, `calcium`, `phosphate`, `magnesium`
- **Renal:** `creatinine`, `cystatin_c`
- **Glucose:** `fasting_glucose`, `glucose_2h`, `hba1c`
- **Lipids:** `total_cholesterol`, `triglyceride`, `hdl_c`, `ldl_c`
- **24h urine:** `urine_na_24h`, `urine_k_24h`, `urine_cl_24h`, `urine_volume_24h`

**Preprocessing and model input:**  
Missing mask **\(R\)** (\(R_j=1\) if feature \(j\) is missing), sample-level missing rate **\(r_{global} = \frac{1}{d}\sum_j R_j\)**, median imputation + z-score normalization → **\(\tilde{C}\)**. Model input: \((x_{ct}, \tilde{C}, R, r_{global})\).

---

## Data & Privacy

- This code was developed using **real patient NCCT and clinical data** under **ethics approval and confidentiality agreements**. **No patient data is distributed with this repository.**
- The repository contains only **method and model code** and non–patient-specific documentation (e.g. this README).
- To reproduce or extend this work, you must:
  - Prepare **CT ROI and clinical tables with the same structure** locally,
  - Obtain **institutional ethics and privacy approvals**,
  - Map your columns to `tumor_type`, `case_id`/`matched_folder`, and the numeric feature columns above.
- Do **not** commit or share **raw CT data, clinical Excel files, or any files that could identify patients** in public repositories or with unauthorized parties.
