"""Evaluate trained models on the leakage-resistant grouped test split.

Ensures no augmented copies of subjects in the test set appeared in the training set.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from animal_injury_detector.config import Config, get_config
from animal_injury_detector.dataset import (
    AnimalInjuryDataset,
    create_leakage_resistant_grouped_splits,
    get_eval_transforms,
    scan_split_directory,
)
from animal_injury_detector.evaluate import evaluate_model
from animal_injury_detector.model import load_checkpoint


def run_grouped_evaluation() -> Dict[str, Any]:
    data_dir = Path("Major/injuries")
    if not data_dir.exists():
        data_dir = Path("dataset/injuries")

    train_samples = scan_split_directory(data_dir / "train")
    val_samples = scan_split_directory(data_dir / "valid")
    test_samples = scan_split_directory(data_dir / "test")

    all_samples = train_samples + val_samples + test_samples

    # Create subject-grouped split with zero subject overlap
    _, _, grouped_test_samples = create_leakage_resistant_grouped_splits(
        all_samples, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    eval_tf = get_eval_transforms(image_size=224)
    test_ds = AnimalInjuryDataset(grouped_test_samples, transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    exp_dirs = sorted(list(Path("results/experiments").glob("exp*")))
    device = torch.device("cpu")
    criterion = nn.BCEWithLogitsLoss()

    grouped_results: Dict[str, Any] = {
        "grouped_test_total_samples": len(grouped_test_samples),
        "grouped_test_healthy": sum(1 for _, l in grouped_test_samples if l == 0),
        "grouped_test_injured": sum(1 for _, l in grouped_test_samples if l == 1),
        "models": {},
    }

    for ed in exp_dirs:
        ckpt_p = ed / "best_model.pth"
        if not ckpt_p.exists():
            continue

        model, metadata = load_checkpoint(ckpt_p, device=device)
        thresh = float(metadata.get("decision_threshold", 0.5))

        metrics, _, _, _ = evaluate_model(
            model=model,
            data_loader=test_loader,
            device=device,
            threshold=thresh,
            criterion=criterion,
        )

        cm = metrics["confusion_matrix"]
        grouped_results["models"][ed.name] = {
            "model_architecture": metadata.get("architecture", "resnet18"),
            "decision_threshold": round(thresh, 4),
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "precision": metrics["precision"],
            "recall_injured": metrics["recall_injured"],
            "false_negatives": cm["false_negatives"],
            "false_positives": cm["false_positives"],
            "true_positives": cm["true_positives"],
            "true_negatives": cm["true_negatives"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "brier_score": metrics["brier_score"],
        }

    out_p = Path("results/leakage_resistant_evaluation.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(grouped_results, f, indent=2)

    print(f"Saved leakage-resistant grouped evaluation to {out_p}")
    print(json.dumps(grouped_results, indent=2))
    return grouped_results


if __name__ == "__main__":
    run_grouped_evaluation()
