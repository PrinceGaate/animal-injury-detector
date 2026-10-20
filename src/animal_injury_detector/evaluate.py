"""Evaluation pipeline for Animal Injury Detector.

Computes precision, recall, F1-score, balanced accuracy, false negatives, confusion matrix,
and threshold optimization with strict clinical priority on injury detection sensitivity.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config, get_config, get_device
from .dataset import create_dataloaders
from .model import load_checkpoint

logger = logging.getLogger(__name__)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    metric: str = "f1",
    candidate_thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Find the optimal decision threshold on a validation dataset.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_probs: Predicted probabilities for the positive class.
        metric: Optimization target ('f1' or 'balanced_acc').
        candidate_thresholds: Array of thresholds to search. Defaults to 0.10 to 0.90.

    Returns:
        Tuple of (best_threshold, best_score).
    """
    if candidate_thresholds is None:
        candidate_thresholds = np.linspace(0.10, 0.90, 81)

    best_thresh = 0.5
    best_score = -1.0

    for t in candidate_thresholds:
        preds = (y_probs >= t).astype(int)
        if metric == "balanced_acc":
            score = float(balanced_accuracy_score(y_true, preds))
        else:  # default "f1"
            score = float(f1_score(y_true, preds, zero_division=0))

        if score > best_score:
            best_score = score
            best_thresh = float(t)

    return best_thresh, best_score


def compute_binary_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute detailed evaluation metrics for binary animal injury classification.

    Follows strict metric reporting order:
    1. Injured Recall
    2. False Negatives (missed injuries)
    3. Precision & False Positive Rate
    4. F1-Score & Balanced Accuracy
    5. Confusion Matrix in order: TP, FN, FP, TN.
    """
    y_pred = (y_probs >= threshold).astype(int)

    # Strictly ordered confusion matrix elements
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    total = len(y_true)
    num_positive = int(np.sum(y_true == 1))
    num_negative = int(np.sum(y_true == 0))

    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if total > 0 else 0.0

    false_negative_rate = float(fn / num_positive) if num_positive > 0 else 0.0
    false_positive_rate = float(fp / num_negative) if num_negative > 0 else 0.0

    # Calibration Brier score (lower is better)
    brier = float(brier_score_loss(y_true, y_probs))

    # Discrimination AUC metrics
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    if len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_probs))
            pr_auc = float(average_precision_score(y_true, y_probs))
        except ValueError:
            pass

    metrics = {
        "total_samples": total,
        "healthy_count": num_negative,
        "injured_count": num_positive,
        "decision_threshold": round(float(threshold), 4),
        "recall_injured": round(recall, 4),
        "false_negatives": fn,
        "false_negative_rate": round(false_negative_rate, 4),
        "precision": round(precision, 4),
        "false_positives": fp,
        "false_positive_rate": round(false_positive_rate, 4),
        "f1_score": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "accuracy": round(accuracy, 4),
        "specificity_healthy": round(specificity, 4),
        "brier_score": round(brier, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "confusion_matrix": {
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "true_negatives": tn,
        },
    }

    return metrics


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    criterion: Optional[nn.Module] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, List[str]]:
    """Evaluate a trained model over a DataLoader using inference mode."""
    model.eval()
    model.to(device)

    all_targets: List[float] = []
    all_probs: List[float] = []
    all_paths: List[str] = []
    running_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for inputs, targets, paths in data_loader:
            inputs = inputs.to(device)
            targets_gpu = targets.to(device).unsqueeze(1)

            logits = model(inputs)

            if criterion is not None:
                loss = criterion(logits, targets_gpu)
                running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            all_probs.extend(probs.tolist())
            all_targets.extend(targets.numpy().tolist())
            all_paths.extend(paths)
            total_samples += inputs.size(0)

    y_true = np.array(all_targets)
    y_probs = np.array(all_probs)

    metrics = compute_binary_metrics(y_true, y_probs, threshold=threshold)
    if criterion is not None and total_samples > 0:
        metrics["loss"] = round(running_loss / total_samples, 6)

    # Statistical sample size warnings
    if metrics["injured_count"] < 15:
        metrics["warning"] = (
            f"Dataset has only {metrics['injured_count']} injured samples. "
            "Metrics (particularly recall and false negatives) carry substantial statistical uncertainty."
        )

    return metrics, y_true, y_probs, all_paths


def run_evaluation(
    checkpoint_path: Union[str, Path],
    data_dir: Optional[Union[str, Path]] = None,
    split: str = "test",
    threshold: Optional[float] = None,
    output_dir: Optional[Union[str, Path]] = None,
    custom_samples: Optional[Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], Optional[List[Tuple[Path, int]]]]] = None,
) -> Dict[str, Any]:
    """Run full evaluation workflow from saved checkpoint."""
    model, metadata = load_checkpoint(checkpoint_path)
    config_dict = metadata.get("config", {})
    config = Config.from_dict(config_dict) if config_dict else get_config()

    if data_dir:
        config.data_dir = Path(data_dir)

    train_loader, val_loader, test_loader, _ = create_dataloaders(config, custom_samples=custom_samples)

    # Use saved optimal threshold if available and not overridden
    chosen_threshold = threshold if threshold is not None else metadata.get("decision_threshold", 0.5)

    target_loader: Optional[DataLoader] = None
    if split == "test":
        target_loader = test_loader
        if target_loader is None:
            logger.warning("No test split found, falling back to validation split.")
            target_loader = val_loader
            split = "valid"
    elif split == "valid":
        target_loader = val_loader
    elif split == "train":
        target_loader = train_loader
    else:
        raise ValueError(f"Unknown split: {split}. Choose from ['train', 'valid', 'test'].")

    device = config.device
    criterion = nn.BCEWithLogitsLoss()
    metrics, _, _, _ = evaluate_model(
        model=model,
        data_loader=target_loader,
        device=device,
        threshold=chosen_threshold,
        criterion=criterion,
    )
    metrics["evaluated_split"] = split
    metrics["checkpoint_path"] = str(checkpoint_path)

    # Print clinical triage summary
    print("\n" + "=" * 65)
    print(f"ANIMAL INJURY DETECTOR - EVALUATION REPORT ({split.upper()} SPLIT)")
    print("=" * 65)
    print(f"Total Samples    : {metrics['total_samples']} (Healthy: {metrics['healthy_count']}, Injured: {metrics['injured_count']})")
    print(f"Decision Cutoff  : {metrics['decision_threshold']}")
    print("-" * 65)
    print(f"1. Injured Recall : {metrics['recall_injured']:.4f}")
    print(f"2. Missed (FN)    : {metrics['false_negatives']} (FN Rate: {metrics['false_negative_rate']:.2%})")
    print(f"3. Precision      : {metrics['precision']:.4f} (FP: {metrics['false_positives']}, FP Rate: {metrics['false_positive_rate']:.2%})")
    print(f"4. F1-Score       : {metrics['f1_score']:.4f}")
    print(f"   Balanced Acc   : {metrics['balanced_accuracy']:.4f}")
    print(f"   Overall Acc    : {metrics['accuracy']:.4f}")
    print(f"   Specificity    : {metrics['specificity_healthy']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"   ROC-AUC        : {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}")
    print("-" * 65)
    print("Confusion Matrix (Ordered: TP, FN, FP, TN):")
    cm = metrics["confusion_matrix"]
    print(f"  [TP] True Positives (Injured correctly detected)  : {cm['true_positives']}")
    print(f"  [FN] False Negatives (Missed Injuries - Danger!)  : {cm['false_negatives']}")
    print(f"  [FP] False Positives (Healthy flagged as Injured) : {cm['false_positives']}")
    print(f"  [TN] True Negatives (Healthy correctly cleared)   : {cm['true_negatives']}")
    print("-" * 65)
    if "warning" in metrics:
        print(f"NOTE: {metrics['warning']}")
    print("=" * 65 + "\n")

    # Save results to json
    out_dir = Path(output_dir or config.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / f"{split}_evaluation.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation report to {report_file}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Animal Injury Detector.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to checkpoint.")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory override.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"], help="Split to evaluate.")
    parser.add_argument("--threshold", type=float, default=None, help="Classification threshold override.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save evaluation report.")
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        threshold=args.threshold,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
