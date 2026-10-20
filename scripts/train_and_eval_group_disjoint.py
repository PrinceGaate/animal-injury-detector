"""Train and evaluate ResNet-18 on the strictly group-disjoint dataset split.

Methodology:
1. Load group-disjoint splits from results/group_disjoint_split_manifest.json (or dataset.py).
2. Validate zero subject prefix overlap and zero sample overlap across train, val, and test.
3. Train ResNet-18 with positive-class weighted BCE on grouped training data only.
4. Model selection (early stopping) and decision threshold selection use validation data ONLY.
5. Evaluate on the held-out grouped test set strictly once after all choices are frozen.
6. Calculate Wilson 95% confidence intervals to honestly convey small-sample statistical uncertainty.
7. Save full metrics, confusion matrix, per-sample predictions, and self-contained checkpoint.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from animal_injury_detector.config import Config, get_config, seed_everything
from animal_injury_detector.dataset import (
    AnimalInjuryDataset,
    create_leakage_resistant_grouped_splits,
    extract_base_prefix,
    get_eval_transforms,
    get_train_transforms,
    scan_split_directory,
)
from animal_injury_detector.evaluate import compute_binary_metrics, evaluate_model, find_optimal_threshold
from animal_injury_detector.model import build_model, load_checkpoint, save_checkpoint
from animal_injury_detector.train import get_loss_criterion, train_model

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute Wilson score interval for binomial proportions (handles small n and boundaries)."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% two-sided normal quantile
    p_hat = float(k / n)
    denominator = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2 * n)) / denominator
    margin = (z / denominator) * math.sqrt((p_hat * (1.0 - p_hat) / n) + ((z**2) / (4 * (n**2))))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return round(lower, 4), round(upper, 4)


def run_pipeline() -> Dict[str, Any]:
    seed_everything(42)

    data_dir = Path("Major/injuries")
    if not data_dir.exists():
        data_dir = Path("dataset/injuries")

    train_samples_raw = scan_split_directory(data_dir / "train")
    val_samples_raw = scan_split_directory(data_dir / "valid")
    test_samples_raw = scan_split_directory(data_dir / "test")
    all_samples = train_samples_raw + val_samples_raw + test_samples_raw

    # 1. Generate strictly group-disjoint splits
    train_samples, val_samples, test_samples = create_leakage_resistant_grouped_splits(
        all_samples, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    tr_prefixes = {extract_base_prefix(p.name) for p, _ in train_samples}
    va_prefixes = {extract_base_prefix(p.name) for p, _ in val_samples}
    te_prefixes = {extract_base_prefix(p.name) for p, _ in test_samples}

    tr_paths = {p.resolve() for p, _ in train_samples}
    va_paths = {p.resolve() for p, _ in val_samples}
    te_paths = {p.resolve() for p, _ in test_samples}

    # Strict assertion of zero leakage
    assert len(tr_prefixes.intersection(va_prefixes)) == 0, "Train-Val prefix overlap detected!"
    assert len(tr_prefixes.intersection(te_prefixes)) == 0, "Train-Test prefix overlap detected!"
    assert len(va_prefixes.intersection(te_prefixes)) == 0, "Val-Test prefix overlap detected!"
    assert len(tr_paths.intersection(va_paths)) == 0, "Train-Val file overlap detected!"
    assert len(tr_paths.intersection(te_paths)) == 0, "Train-Test file overlap detected!"
    assert len(va_paths.intersection(te_paths)) == 0, "Val-Test file overlap detected!"

    print("======================================================================")
    print("LEAKAGE-FREE GROUP-DISJOINT SPLIT AUDIT PASSED")
    print(f"Train samples: {len(train_samples)} (Healthy: {sum(1 for _, l in train_samples if l==0)}, Injured: {sum(1 for _, l in train_samples if l==1)}) in {len(tr_prefixes)} groups")
    print(f"Val samples:   {len(val_samples)} (Healthy: {sum(1 for _, l in val_samples if l==0)}, Injured: {sum(1 for _, l in val_samples if l==1)}) in {len(va_prefixes)} groups")
    print(f"Test samples:  {len(test_samples)} (Healthy: {sum(1 for _, l in test_samples if l==0)}, Injured: {sum(1 for _, l in test_samples if l==1)}) in {len(te_prefixes)} groups")
    print("======================================================================")

    # 2. Configure training
    config = get_config(
        experiment_name="group_disjoint_resnet18_weighted_bce",
        model_name="resnet18",
        loss_type="weighted_bce",
        checkpoint_dir=Path("results/experiments/group_disjoint_resnet18_weighted_bce"),
        best_checkpoint_name="best_model.pth",
        epochs=4,
        batch_size=32,
        learning_rate=3e-4,
        dropout_rate=0.2,
        pretrained=True,
        freeze_backbone_epochs=0,
        optimize_threshold_val=True,
        random_seed=42,
    )
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    summary_file = config.checkpoint_dir / "training_summary.json"
    if not config.best_checkpoint_path.exists() or not summary_file.exists():
        # 3. Train using ONLY train_samples and val_samples
        # We pass (train_samples, val_samples, None) so the trainer never sees test_samples
        trained_model, training_results = train_model(
            config=config,
            custom_samples=(train_samples, val_samples, None),
        )
    else:
        logger.info("Loading existing trained group-disjoint model from %s", config.best_checkpoint_path)
        with open(summary_file, "r", encoding="utf-8") as f:
            training_results = json.load(f)

    best_val_f1 = training_results["best_val_f1"]
    best_epoch = training_results["best_epoch"]
    val_selected_threshold = float(training_results.get("selected_threshold", 0.5))
    print(f"\nTraining State: Best Val F1 = {best_val_f1:.4f} at Epoch {best_epoch} (Validation-Selected Threshold = {val_selected_threshold:.4f})")

    # 4. Load the best validation checkpoint
    best_model, metadata = load_checkpoint(config.best_checkpoint_path, device=torch.device("cpu"))

    # 5. Evaluate on Held-Out Grouped Test Set strictly once
    eval_tf = get_eval_transforms(image_size=config.image_size)
    test_ds = AnimalInjuryDataset(test_samples, transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()

    # Evaluation at default threshold 0.50
    metrics_05, y_test_true, y_test_probs, test_paths = evaluate_model(
        model=best_model,
        data_loader=test_loader,
        device=torch.device("cpu"),
        threshold=0.50,
        criterion=criterion,
    )

    # Evaluation at validation-calibrated threshold
    metrics_calibrated, _, _, _ = evaluate_model(
        model=best_model,
        data_loader=test_loader,
        device=torch.device("cpu"),
        threshold=val_selected_threshold,
        criterion=criterion,
    )

    # Compute Wilson Confidence Intervals on the calibrated metrics
    cal_cm = metrics_calibrated["confusion_matrix"]
    tp, fn, fp, tn = cal_cm["true_positives"], cal_cm["false_negatives"], cal_cm["false_positives"], cal_cm["true_negatives"]
    num_pos = tp + fn
    num_neg = tn + fp

    rec_ci = wilson_score_interval(tp, num_pos)
    prec_ci = wilson_score_interval(tp, tp + fp)
    spec_ci = wilson_score_interval(tn, num_neg)
    acc_ci = wilson_score_interval(tp + tn, len(y_test_true))

    metrics_calibrated["wilson_95_confidence_intervals"] = {
        "recall_injured_95_ci": list(rec_ci),
        "precision_95_ci": list(prec_ci),
        "specificity_healthy_95_ci": list(spec_ci),
        "accuracy_95_ci": list(acc_ci),
        "statistical_uncertainty_note": (
            f"The test set contains {num_pos} injured examples out of {len(y_test_true)} total. "
            f"The 95% Wilson confidence interval for injured recall ({metrics_calibrated['recall_injured']*100:.1f}%) "
            f"ranges from {rec_ci[0]*100:.1f}% to {rec_ci[1]*100:.1f}%. "
            "Because clinical conclusions require high certainty, this small test sample warrants caution."
        ),
    }

    # Format per-sample predictions
    per_image_preds = []
    for idx, (path_str, true_lbl, prob) in enumerate(zip(test_paths, y_test_true, y_test_probs)):
        pred_cal = int(prob >= val_selected_threshold)
        per_image_preds.append({
            "image_path": str(Path(path_str).relative_to(Path("Major").parent)),
            "filename": Path(path_str).name,
            "prefix": extract_base_prefix(Path(path_str).name),
            "ground_truth": int(true_lbl),
            "ground_truth_label": "injured" if true_lbl == 1 else "healthy",
            "injury_probability": round(float(prob), 4),
            "predicted_class_calibrated": "injured" if pred_cal == 1 else "healthy",
            "predicted_label_calibrated": pred_cal,
            "correct": bool(pred_cal == true_lbl),
        })

    # Save outputs
    eval_summary = {
        "evaluation_protocol": "strictly_group_disjoint",
        "random_seed": 42,
        "dataset_split_counts": {
            "train_samples": len(train_samples),
            "train_healthy": sum(1 for _, l in train_samples if l==0),
            "train_injured": sum(1 for _, l in train_samples if l==1),
            "val_samples": len(val_samples),
            "val_healthy": sum(1 for _, l in val_samples if l==0),
            "val_injured": sum(1 for _, l in val_samples if l==1),
            "test_samples": len(test_samples),
            "test_healthy": sum(1 for _, l in test_samples if l==0),
            "test_injured": sum(1 for _, l in test_samples if l==1),
        },
        "model_architecture": "resnet18",
        "loss_type": "weighted_bce",
        "trained_epochs": best_epoch,
        "validation_selected_threshold": val_selected_threshold,
        "validation_best_f1": best_val_f1,
        "test_results_default_threshold_0.5": metrics_05,
        "test_results_calibrated_threshold": metrics_calibrated,
    }

    out_eval_path = Path("results/group_disjoint_evaluation.json")
    with open(out_eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    out_preds_path = Path("results/group_disjoint_test_predictions.json")
    with open(out_preds_path, "w", encoding="utf-8") as f:
        json.dump(per_image_preds, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL GROUP-DISJOINT EVALUATION ON HELD-OUT TEST DATA")
    print("=" * 70)
    print(f"Total Test Samples : {len(test_samples)} ({sum(1 for _, l in test_samples if l==0)} Healthy, {sum(1 for _, l in test_samples if l==1)} Injured)")
    print(f"Decision Threshold : {val_selected_threshold:.4f} (Calibrated on Validation Data ONLY)")
    print(f"Accuracy           : {metrics_calibrated['accuracy']*100:.2f}% (95% CI: [{acc_ci[0]*100:.1f}%, {acc_ci[1]*100:.1f}%])")
    print(f"Balanced Accuracy  : {metrics_calibrated['balanced_accuracy']*100:.2f}%")
    print(f"Injured Recall     : {metrics_calibrated['recall_injured']*100:.2f}% ({tp}/{num_pos}) (95% CI: [{rec_ci[0]*100:.1f}%, {rec_ci[1]*100:.1f}%])")
    print(f"Missed (FN)        : {fn} (False Negative Rate: {metrics_calibrated['false_negative_rate']*100:.2f}%)")
    print(f"Precision          : {metrics_calibrated['precision']*100:.2f}% (95% CI: [{prec_ci[0]*100:.1f}%, {prec_ci[1]*100:.1f}%])")
    print(f"False Alarms (FP)  : {fp} (False Positive Rate: {metrics_calibrated['false_positive_rate']*100:.2f}%)")
    print(f"Specificity        : {metrics_calibrated['specificity_healthy']*100:.2f}%")
    print(f"F1-Score           : {metrics_calibrated['f1_score']:.4f}")
    print(f"Brier Score        : {metrics_calibrated['brier_score']:.4f} (Probability quality metric)")
    print(f"Confusion Matrix   : TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    print("=" * 70)

    return eval_summary


if __name__ == "__main__":
    run_pipeline()
