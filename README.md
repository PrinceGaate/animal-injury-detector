# Animal Injury Detector

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A clean, reproducible, tested, and portfolio-ready deep learning pipeline for binary classification of animal injuries. Built with PyTorch and torchvision, this repository enables automated screening of animal photos (stray dogs, cats, wildlife) to detect visible physical injuries and prioritize urgent veterinary triage.

---

## 1. Project Overview

Animal rescue organizations, shelters, and mobile triage teams handle hundreds of citizen-submitted rescue reports daily. Rapidly screening these images allows rescue coordinators to prioritize animals suffering from open wounds, trauma, or visible injuries over animals requiring routine intake.

This project refactors an exploratory autoencoder notebook into a modular, production-grade Python package with deterministic training, class-imbalance compensation, data leakage audits, self-describing checkpoint artifacts, and evaluation metrics tailored for high-sensitivity injury triage.

---

## 2. Problem Statement & Intended Use

### The Problem
In animal rescue triage, **false negatives are catastrophic**. A missed injury (classifying a wounded animal as healthy) leaves an animal to suffer untreated infection or trauma. Conversely, false positives (flagging a healthy animal as injured) merely prompt a secondary photo review by a volunteer. Standard accuracy is an inappropriate metric because healthy animals typically outnumber injured animals.

### Intended Use
- **Field Triage Support**: Assisting rescue coordinators in queue prioritization.
- **Citizen Intake Pre-filtering**: Highlighting suspected physical injuries in community rescue reports.

> [!CAUTION]
> **Veterinary Screening Disclaimer**: This software is an engineering decision-support tool. It is **NOT** a certified veterinary diagnostic medical device. It cannot detect internal trauma, subcutaneous fractures, systemic illness, or non-visible distress, and must never replace examination by a licensed veterinarian.

---

## 3. Features & Architecture

### Supervised Classification Pipeline
- **Backbone**: Feature extraction using ResNet-18 residual blocks pretrained on ImageNet-1K (`ResNet18_Weights.DEFAULT`).
- **Classification Head**: Adaptive average pooling $\rightarrow$ Dropout ($p=0.2$) $\rightarrow$ Linear layer ($\text{in\_features}=512 \rightarrow 1$).
- **Output**: Single raw logit $z$. The predicted probability is calculated via Sigmoid: $P(\text{injured}) = \sigma(z) = \frac{1}{1 + e^{-z}}$.
- **Loss**: `nn.BCEWithLogitsLoss(pos_weight \approx 6.01)` to penalize false negatives on the minority injured class.

```
Input Image (3 x 224 x 224)
        │
┌───────▼────────────────────────┐
│   ResNet-18 Backbone (Conv)    │
│   512-dim Feature Representation│
└───────┬────────────────────────┘
        │
┌───────▼────────────────────────┐
│  Dropout (p=0.2) + Linear(512,1)│
└───────┬────────────────────────┘
        │
    Raw Logit ──► Sigmoid ──► Probability of Injury [0.0 - 1.0]
```

### Contrast with Historical Autoencoder (`notebooks/Major.ipynb`)
The original notebook (`notebooks/Major.ipynb`) implemented an **unsupervised convolutional autoencoder** trained solely on 105 injured images to flag anomalies by reconstruction Mean Squared Error (MSE). In contrast, the inference script expected a supervised classifier. The historical autoencoder is preserved in `notebooks/Major.ipynb` for research lineage, while `src/animal_injury_detector` implements the supervised binary classifier (`healthy = 0`, `injured = 1`).

---

## 4. Dataset Source, Class Mapping, & Data Leakage Audit

### Data Source & Class Mapping
- **Source**: Sourced from Roboflow Universe (`https://universe.roboflow.com/project-ienbb/injured-animal`), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Class Mapping**:
  - `injuries` $\rightarrow$ Positive Class Label `1`
  - `Healthy` $\rightarrow$ Negative Class Label `0`

### Data Leakage Resolution: Strictly Group-Disjoint Splits
Audit of the raw dataset revealed that Roboflow geometric augmentations (rotations, flips, crops) were generated *prior* to dataset splitting:
- In the original splits, **87 animal prefixes in train appeared in valid**, and **47 appeared in test**.
- In addition, an initial post-hoc grouped evaluation was contaminated because the tested models had been trained on the original training set, exposing them to **90 of the 121 images** in that test set during training.

To guarantee true out-of-distribution evaluation, we partitioned all 802 images into strictly group-disjoint splits **before training**:

| Split | Healthy ($0$) | Injured ($1$) | Total Images | Distinct Groups | Overlap with Other Splits |
|---|:---:|:---:|:---:|:---:|:---:|
| **Train** | 469 | 78 | 547 | 279 | **0** |
| **Validation** | 107 | 15 | 122 | 59 | **0** |
| **Test** | 124 | 9 | 133 | 59 | **0** |
| **Total** | **700** | **102** | **802** | **397** | **0** |

*Split manifests and group identifiers are saved in [`results/group_disjoint_split_manifest.json`](results/group_disjoint_split_manifest.json).*

> [!NOTE]
> **Methodological Note on Prefix Grouping**: This partition is strictly **prefix-disjoint**; it guarantees that no file sharing a base prefix string or Roboflow augmented variant crosses splits. However, web-scraped filenames alone cannot definitively prove whether two differently named files depict the same underlying animal. Therefore, this split is rigorously prefix-disjoint, rather than proven subject-independent.

---

## 5. Repository Structure

```
animal-injury-detector/
├── notebooks/
│   └── Major.ipynb                 # Preserved historical autoencoder experiment
├── src/
│   └── animal_injury_detector/
│       ├── __init__.py             # Package exports (v1.0.0)
│       ├── config.py               # Hyperparameters, paths, and portable device setup
│       ├── dataset.py              # Transforms, class mapping, and group-disjoint splitting
│       ├── model.py                # ResNet-18 baseline, checkpoint serialization
│       ├── train.py                # Training loop with class weighting and early stopping
│       ├── infer.py                # Single image and batch directory inference CLI
│       └── evaluate.py             # Sensitivity metrics, confusion matrix, threshold search
├── dataset/
│   └── README.md                   # In-depth dataset documentation and setup guide
├── checkpoints/
│   ├── .gitkeep                    # Preserves directory in git
│   └── best_model.pth              # Retained verified model checkpoint (tau* = 0.7300)
├── results/
│   ├── comparative_benchmark_report.md     # Full engineering benchmark report
│   ├── dataset_leakage_audit.json          # Subject prefix leakage breakdown across splits
│   ├── group_disjoint_split_manifest.json  # Manifest of groups and sample assignments
│   ├── group_disjoint_evaluation.json      # Full metrics on held-out group-disjoint test set
│   ├── group_disjoint_test_predictions.json# Per-image test predictions and confidence scores
│   └── training_history.json               # Epoch-by-epoch loss and validation metrics
├── scripts/
│   ├── audit_prefixes.py                   # Prefix extraction and label consistency audit
│   ├── leakage_analysis.py                 # Split leakage and manifest generator
│   ├── run_experiments.py                  # 5-experiment controlled benchmark runner
│   └── train_and_eval_group_disjoint.py    # Group-disjoint training and evaluation pipeline
├── tests/
│   ├── test_dataset.py             # Unit tests for transforms, loading, and class mapping
│   └── test_model.py               # Unit tests for forward pass, checkpoints, and inference
├── .gitignore                      # Excludes binaries, raw datasets, caches, and weights
├── requirements.txt                # Lean dependency specification
├── pyproject.toml                  # Standard package configuration and CLI entrypoints
├── pytest.ini                      # Test runner configuration (pythonpath = src)
├── README.md                       # Repository documentation
└── LICENSE                         # MIT Open Source License
```

---

## 6. Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/animal-injury-detector.git
cd animal-injury-detector
```

### 2. Create and Activate a Virtual Environment
```bash
# On Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

---

## 7. Training

### Train on Group-Disjoint Splits
To train the finalized ResNet-18 model on strictly group-disjoint data (model selection and threshold tuning use validation data ONLY):

```bash
python scripts/train_and_eval_group_disjoint.py
```

### General Training CLI
```bash
python -m animal_injury_detector.train \
  --model-name resnet18 \
  --loss-type weighted_bce \
  --epochs 10 \
  --batch-size 32 \
  --lr 3e-4 \
  --patience 3
```

---

## 8. Inference

Generate predictions using `infer.py` with automatic calibrated threshold loading:

### Single Image Prediction
```bash
python -m animal_injury_detector.infer \
  --checkpoint checkpoints/best_model.pth \
  --image Major/injuries/test/injuries/cat-5-_png.rf.2d0e03b4bab14db78d4c8914cbc45140.jpg
```

**Example Output:**
```text
=======================================================
ANIMAL INJURY PREDICTION RESULT
=======================================================
Image       : Major/injuries/test/injuries/cat-5-_png.rf.2d0e03b4bab14db78d4c8914cbc45140.jpg
Model       : resnet18
Prediction  : INJURED
Score       : 95.35%
Threshold   : 0.73
Note        : Model estimation only. NOT a certified veterinary diagnostic certainty.
=======================================================
```

### Batch Directory Prediction
```bash
python -m animal_injury_detector.infer \
  --checkpoint checkpoints/best_model.pth \
  --image-dir path/to/unlabeled_images/ \
  --output results/batch_predictions.json
```

---

## 9. Evaluation & Verified Benchmark Results

### Table 1: Final Performance on Strictly Group-Disjoint Test Set (133 samples: 124 Healthy, 9 Injured)

Evaluated on 59 distinct prefix groups with **zero training or validation exposure** (strictly prefix-disjoint):

| Metric | Default Cutoff ($\tau = 0.50$) | Calibrated Cutoff ($\tau^* = 0.7300$) | 95% Wilson Confidence Interval ($\tau^*$) |
|---|:---:|:---:|:---:|
| **Accuracy** | 90.23% | **97.74%** | [93.6%, 99.2%] |
| **Balanced Accuracy** | 89.61% | **93.64%** | — |
| **Injured Recall** | **88.89%** | **88.89%** | **[56.5%, 98.0%]** |
| **Missed Injuries (FN)** | **1** | **1** | — |
| **False Negative Rate** | 11.11% | **11.11%** | — |
| **Precision** | 40.00% | **80.00%** | **[49.0%, 94.3%]** |
| **False Alarms (FP)** | 12 | **2** | — |
| **False Positive Rate** | 9.68% | **1.61%** | — |
| **Specificity** | 90.32% | **98.39%** | [94.3%, 99.6%] |
| **F1-Score** | 0.5517 | **0.8421** | — |
| **Brier Score** | 0.0748 | **0.0748** | — |
| **ROC-AUC** | 0.9462 | **0.9462** | — |
| **PR-AUC** | 0.9034 | **0.9034** | — |

### Confusion Matrix ($\tau^* = 0.7300$)
```
                     Predicted Healthy    Predicted Injured
Actual Healthy (124)       122 (TN)               2 (FP)
Actual Injured (9)           1 (FN)               8 (TP)
```

> [!WARNING]
> **Small-Sample Statistical Uncertainty**: The test set contains **9 injured examples** out of 133 total images. The 95% Wilson confidence interval for injured recall spans **[56.5%, 98.0%]**, and for precision spans **[49.0%, 94.3%]**. While the model caught 8 of 9 injuries with only 2 false alarms, small sample sizes warrant caution and clinical verification.

---

## 10. Historical Benchmark Comparison (Original Roboflow Split)

All 5 controlled baseline experiments on the original split (`seed = 42`) are preserved in [`results/experiments_summary.json`](results/experiments_summary.json):

| Experiment ID | Architecture | Imbalance Strategy | Val Threshold $\tau^*$ | Test Accuracy | Balanced Acc | Injured Recall | Missed (FN) | False Alarms (FP) | Precision | F1-Score | Brier Score |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exp 1** | ResNet-18 | Baseline Unweighted BCE | 0.33 | 97.44% | 88.89% | 77.78% | 2 | 0 | 100.0% | 0.8750 | 0.0531 |
| **Exp 2** | ResNet-18 | Weighted BCE ($\text{pos\_weight}=5.86$) | 0.67 | 97.44% | 88.89% | 77.78% | 2 | 0 | 100.0% | 0.8750 | 0.0972 |
| **Exp 3** | ResNet-18 | WeightedRandomSampler | 0.58 | 97.44% | 88.89% | 77.78% | 2 | 0 | 100.0% | 0.8750 | 0.0718 |
| **Exp 4** | ResNet-18 | Advanced Aug + 2-Stage Warmup | 0.65 | 97.44% | 88.89% | 77.78% | 2 | 0 | 100.0% | 0.8750 | 0.1438 |
| **Exp 5** | EfficientNet-B0 | Weighted BCE + 2-Stage Warmup | 0.58 | 92.31% | 85.99% | 77.78% | 2 | 4 | 63.64% | 0.7000 | 0.2332 |

---

## 11. Automated Verification & Unit Tests

Run the offline unit test suite with `pytest`:

```bash
python -m pytest -v
```

All 14 tests run locally without GPU or internet connectivity in ~7.5 seconds:
- `tests/test_dataset.py`: 6 tests passing (class mapping, prefix extraction, transforms output shape, synthetic dataset loading, weighted sampler, leakage-resistant grouping).
- `tests/test_model.py`: 8 tests passing (ResNet/EfficientNet/MobileNet forward passes, backbone freezing, checkpoint roundtrip serialization, autoencoder rejection, missing checkpoint error handling, predictor pipeline, binary metrics calculation, validation threshold search).

---

## 12. Limitations & Future Work

1. **Small Positive Sample Size**: Evaluating on 9 independent injured test instances yields wide confidence intervals ([56.5%, 98.0%]). Ongoing data collection from rescue partners is needed to expand the positive test set to 100+ cases.
2. **Injury Typology**: Current models perform binary triage. Future iterations will classify injury severity and type (laceration, puncture, mange, fractures, infection).
3. **Lesion Localization**: Integrating object detection (YOLOv8/v11) will enable visual bounding boxes around wounds to assist veterinarians in rapid verification.

---

## 13. License & Attribution

- **Software Code**: [MIT License](LICENSE) &copy; 2026 Animal Injury Detector Contributors.
- **Dataset Attribution**: Sourced from Roboflow Universe under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
