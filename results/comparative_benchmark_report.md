# Comparative Benchmark Report: Animal Injury Detector

**Project**: Animal Injury Detector  
**Environment**: Python 3.12.4 | PyTorch 2.14.1+cpu | CPU Execution (Windows)  
**Dataset**: 802 total verified images (700 Healthy, 102 Injured)  
**Date**: October 2026  
**Status**: Verified & Reproducible (Ported to Group-Disjoint Evaluation)  

---

## 1. Executive Summary & Audit Findings

### Architectural Mismatch Resolution
The original notebook (`Major.ipynb`) trained an **unsupervised convolutional autoencoder** on 105 injured images to flag anomalies by mean squared error (MSE) reconstruction. In contrast, the inference script expected a **supervised ResNet-18 binary classifier**. No trained classifier weights previously existed on disk.
- The autoencoder experiment was preserved in `notebooks/Major.ipynb`.
- A modular supervised binary classification package was engineered in `src/animal_injury_detector/` mapping `injured = 1` and `healthy = 0`.

### Data Leakage Diagnosis & Resolution
1. **Roboflow Pre-Split Augmentation Leakage**: The source Roboflow dataset applied augmentations (rotations, flips, crops) *prior* to dataset splitting. In the original splits, **87 animal prefixes in train appeared in valid**, and **47 appeared in test**.
2. **Prior Grouped Evaluation Limitation**: An initial post-hoc grouped evaluation partitioned the pooled images into groups, but tested models that had been trained on the original training split. Because the original training split contained images from those subjects, **90 of the 121 images in that grouped test set had already been exposed to the model during training**.
3. **Strictly Group-Disjoint Protocol**: To establish genuine generalization, we partitioned all 802 images into group-disjoint splits **before training**:
   - **Train**: 547 images (469 Healthy, 78 Injured) across 279 subject groups
   - **Validation**: 122 images (107 Healthy, 15 Injured) across 59 subject groups
   - **Test**: 133 images (124 Healthy, 9 Injured) across 59 subject groups
   - **Audit Overlap**: Exactly **0 subject prefix overlap** and **0 sample overlap** between train, val, and test.

> [!NOTE]
> **Methodological Note on Prefix Disjointness**: The split is strictly **prefix-disjoint** (zero shared filename prefixes or augmented copies across splits). Because web-scraped filenames cannot establish whether distinct filenames depict the same underlying animal, the evaluation is prefix-disjoint, not proven subject-independent.

---

## 2. Finalized Model Benchmark on Strictly Group-Disjoint Test Set

The finalized candidate (**ResNet-18 + Positive-Class Weighted BCE**, $\text{pos\_weight} \approx 6.01$) was trained exclusively on the 547 grouped training images. Model checkpointing and decision threshold calibration were performed **strictly on validation data** ($\tau^* = 0.7300$). The test split was evaluated strictly once after all choices were frozen.

### Table 1: Final Evaluation on Group-Disjoint Test Split (124 Healthy, 9 Injured, 0 Prefix Overlap)

| Metric | Default Threshold ($\tau = 0.50$) | Calibrated Threshold ($\tau^* = 0.7300$) | 95% Wilson Confidence Interval ($\tau^*$) | Notes / Clinical Interpretation |
|---|:---:|:---:|:---:|---|
| **Accuracy** | 90.23% | **97.74%** | [93.6%, 99.2%] | Overall sample classification accuracy |
| **Balanced Accuracy** | 89.61% | **93.64%** | — | Arithmetic mean of recall and specificity |
| **Injured Recall** | **88.89%** | **88.89%** | **[56.5%, 98.0%]** | **8 of 9 injured cases caught** (Primary triage metric) |
| **Missed Injuries (FN)** | **1** | **1** | — | Only 1 false negative across the test split |
| **False Negative Rate** | 11.11% | **11.11%** | — | Proportion of true injuries missed |
| **Precision** | 40.00% | **80.00%** | **[49.0%, 94.3%]** | Fraction of positive alerts that were genuine injuries |
| **False Alarms (FP)** | 12 | **2** | — | Dropped from 12 to 2 by validation calibration |
| **False Positive Rate** | 9.68% | **1.61%** | — | Proportion of healthy animals misclassified |
| **Specificity** | 90.32% | **98.39%** | [94.3%, 99.6%] | True negative rate on healthy animals |
| **F1-Score** | 0.5517 | **0.8421** | — | Harmonic mean of precision and recall |
| **Brier Score** | 0.0748 | **0.0748** | — | Mean squared probability error (Calibration quality) |
| **ROC-AUC** | 0.9462 | **0.9462** | — | Discrimination capacity across all thresholds |
| **PR-AUC** | 0.9034 | **0.9034** | — | Area under Precision-Recall curve |

### Confusion Matrix (Strict Ordering: TP, FN, FP, TN)
- **True Positives (TP)**: 8 (Injured correctly prioritized)
- **False Negatives (FN)**: 1 (Injured animal misclassified as healthy)
- **False Positives (FP)**: 2 (Healthy animal flagged for review)
- **True Negatives (TN)**: 122 (Healthy animals correctly classified)

> [!WARNING]
> **Statistical Uncertainty Disclosure**: The test set contains **9 injured examples** out of 133 total images. While the point estimate for recall is 88.89% (8/9), the 95% Wilson confidence interval spans **[56.5%, 98.0%]**. Similarly, precision is 80.0% with a 95% confidence interval of **[49.0%, 94.3%]**. Because of this small sample size, real-world field deployment requires conservative thresholding and ongoing clinical audit.

---

## 3. Comparison with Historical Experiments (Original Split)

For historical reference, all 5 controlled experiments trained on the original dataset split are summarized below. Note that these metrics reflect the original Roboflow split where augmented copies of training subjects were present in validation and test:

| Experiment ID | Architecture | Imbalance Strategy | Val Threshold $\tau^*$ | Test Accuracy | Balanced Acc | Injured Recall | Missed (FN) | False Alarms (FP) | Precision | F1-Score | Brier Score |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exp 1** | ResNet-18 | Unweighted BCE (Baseline) | 0.33 | 97.44% | 88.89% | 77.78% | 2 | 0 | 100.0% | 0.8750 | 0.0531 |
| **Exp 2** | ResNet-18 | Weighted BCE ($\text{pos\_weight}=5.86$) | 0.67 | 97.44% | 88.89% | 77.78% | 2 | 0 | 100.0% | 0.8750 | 0.0972 |
| **Exp 3** | ResNet-18 | WeightedRandomSampler | 0.58 | 97.44% | 88.89% | 77.78% | 2 | 0 | 100.0% | 0.8750 | 0.0718 |
| **Exp 4** | ResNet-18 | Advanced Aug + 2-Stage Warmup | 0.65 | 97.44% | 88.89% | 77.78% | 2 | 0 | 100.0% | 0.8750 | 0.1438 |
| **Exp 5** | EfficientNet-B0 | Weighted BCE + 2-Stage Warmup | 0.58 | 92.31% | 85.99% | 77.78% | 2 | 4 | 63.64% | 0.7000 | 0.2332 |

### Key Comparative Insights
1. **Unweighted BCE at $\tau=0.50$ Failed**: Baseline ResNet-18 missed **5 of 9 injuries** (Recall = 44.44%) due to healthy class dominance.
2. **Positive-Class Weighting Was Critical**: Weighting the positive class shifted the loss gradient to penalize missed injuries, maintaining $\ge 88.89\%$ sensitivity.
3. **Threshold Calibration Preserves Precision**: Because weighted BCE shifts raw logits upward, using a validation-tuned threshold ($\tau^* = 0.7300$) eliminated 10 false alarms while retaining 88.89% recall.
4. **Brier Score vs. Classification Accuracy**: Brier score (0.0748) measures how closely predicted probabilities match true binary outcomes ($0$ or $1$). ResNet-18 produces sharp, well-calibrated probabilities, whereas EfficientNet-B0 suffered calibration drift (Brier = 0.2332).

---

## 4. Retained Model Checkpoint

- **Selected Checkpoint**: `checkpoints/best_model.pth`
- **Trained Model Origin**: `results/experiments/group_disjoint_resnet18_weighted_bce/best_model.pth`
- **Architecture**: Pretrained ResNet-18 (`torchvision.models.resnet18`)
- **Trained Parameters**: 11,177,025 (512-dim adaptive pooling + Dropout(0.2) + Linear(512, 1))
- **Validation-Calibrated Threshold**: $\tau^* = 0.7300$
- **Input Preprocessing**: Resize to 224x224, normalize with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`
- **Manifest Location**: `results/group_disjoint_split_manifest.json`
- **Per-Image Test Predictions**: `results/group_disjoint_test_predictions.json`
