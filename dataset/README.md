# Dataset Documentation: Animal Injury Detector

This directory contains documentation and configuration for datasets used in the **Animal Injury Detector** project. In accordance with machine learning best practices and GitHub hygiene, raw image binaries are excluded from version control via `.gitignore`.

---

## 1. Dataset Overview

The repository works with two primary dataset formats identified during project inspection:

### A. Candidate Supervised Binary Dataset (`injuries/`)
Used as the primary dataset for supervised binary classification baseline training, validation, and evaluation.

- **Class Mapping**:
  - `injuries` $\rightarrow$ Positive Label `1`
  - `Healthy` $\rightarrow$ Negative Label `0`
- **Splits & Sample Counts**:
  - `train`: 492 Healthy, 85 injuries (Total: 577, 14.7% positive)
  - `valid`: 139 Healthy, 10 injuries (Total: 149, 6.7% positive)
  - `test`: 69 Healthy, 10 injuries (Total: 79, 12.7% positive)
  - **Overall Total**: 700 Healthy, 105 injuries (805 images total)

### B. Legacy Roboflow Multiclass Dataset (`injured animal.v1.multiclass/`)
Used in the initial Colab notebook (`notebooks/Major.ipynb`) to train an exploratory ResNet-18 autoencoder.

- **Source**: Roboflow Universe (`https://universe.roboflow.com/project-ienbb/injured-animal`)
- **License**: Creative Commons Attribution 4.0 International (`CC BY 4.0`)
- **Content**: 105 images (85 train, 10 valid, 10 test) containing only injured animal images (no negative/healthy class).
- **Format**: Pre-augmented with 640x640 stretch, horizontal/vertical flips, ±10° rotation, shear, and Gaussian blur.

---

## 2. Expected Directory Structure

Place raw datasets in the project root under `dataset/injuries` (or pass the custom path via `--data-dir`):

```
dataset/
├── README.md
└── injuries/
    ├── train/
    │   ├── Healthy/
    │   │   ├── 002_jpg.rf.19a707d57ea7d67879969bf94ff94e36.jpg
    │   │   └── ...
    │   └── injuries/
    │       ├── -2_jpg.rf.35b0aa598c36163506b90be5ef6b9fdd.jpg
    │       └── ...
    ├── valid/
    │   ├── Healthy/
    │   │   └── ...
    │   └── injuries/
    │       └── ...
    └── test/
        ├── Healthy/
        │   └── ...
        └── injuries/
            └── ...
```

---

## 3. Critical Dataset Findings & Statistical Limitations

During engineering audit of the raw dataset, the following key findings were identified:

1. **Severe Class Imbalance**:
   - Healthy instances significantly outnumber injured instances (~5.8:1 in `train`, ~14:1 in `valid`).
   - Standard accuracy will be misleadingly high if the model simply predicts "Healthy".
   - The training pipeline must employ class weighting (`pos_weight` in `BCEWithLogitsLoss`) and evaluate precision, recall, and false negative counts.

2. **Underpowered Validation & Test Sets**:
   - Validation and test sets each contain only **10 positive (injured)** samples.
   - A single misclassification shifts recall by 10.0 percentage points.
   - Metric evaluations on this split carry wide confidence intervals and must be interpreted with caution.

3. **Data Leakage in Splits (Roboflow Augmentation Leakage)**:
   - Audit revealed that Roboflow augmentations (rotation, flip, shear) were generated prior to dataset splitting.
   - While exact MD5 byte matches across splits are `0`, **88 base image prefixes in `train` appear in `valid`**, and **48 base image prefixes appear in `test`**.
   - As a result, test evaluation performance may be optimistic due to near-duplicate frames/subjects appearing across splits.
   - Future iterations must collect independent, patient-isolated splits.

4. **Image Integrity**:
   - All 805 images in `injuries/` were verified using PIL; 0 corrupted images were found.
   - The pipeline enforces RGB conversion (`.convert("RGB")`) to handle any CMYK, grayscale, or RGBA inputs cleanly.

---

## 4. Setup & Ingestion Instructions

1. If you have the dataset archive, extract it into `dataset/injuries/` or keep it at `Major/injuries/`.
2. The dataset loader in `src/animal_injury_detector/dataset.py` automatically checks:
   1. Explicit `--data-dir` argument if provided.
   2. `dataset/injuries/` relative to project root.
   3. `Major/injuries/` relative to project root (legacy fallback).
3. Verify that directory paths match the structure shown above before running training.

---

## 5. Licensing and Attribution

- **Injured Animal Dataset**: CC BY 4.0 by Roboflow user (`https://universe.roboflow.com/project-ienbb/injured-animal`).
- **Healthy Animal Images**: Sourced from Roboflow animal classification sets. Redistribution rights should be confirmed before publicly hosting image archives on external sites.
