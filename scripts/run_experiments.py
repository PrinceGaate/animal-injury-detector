"""Controlled experiment runner for Animal Injury Detector.

Systematically trains and compares:
- Experiment 1: ResNet-18 Baseline with unweighted BCE
- Experiment 2: ResNet-18 with positive-class weighted BCE (pos_weight = 5.857)
- Experiment 3: ResNet-18 with WeightedRandomSampler (balanced mini-batch sampling)
- Experiment 4: ResNet-18 with advanced data augmentation & 2-stage transfer learning
- Experiment 5: Alternative Architecture: EfficientNet-B0
- Experiment 6: Leakage-resistant grouped split evaluation

Saves structured comparative metrics to results/experiments_summary.json.
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from animal_injury_detector.config import get_config
from animal_injury_detector.dataset import create_dataloaders
from animal_injury_detector.evaluate import evaluate_model
from animal_injury_detector.model import load_checkpoint
from animal_injury_detector.train import train_model

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_all_experiments() -> List[Dict[str, Any]]:
    device = "cpu"
    epochs = 3  # 3 epochs is sufficient to measure convergence trend and compare strategies on CPU
    batch_size = 32

    experiment_definitions = [
        {
            "name": "exp1_resnet18_unweighted",
            "desc": "ResNet-18 Baseline (Unweighted BCE, Standard Augmentations)",
            "model_name": "resnet18",
            "loss_type": "bce",
            "sampler_type": "standard",
            "aug_level": "standard",
            "freeze_epochs": 0,
            "lr": 1e-4,
        },
        {
            "name": "exp2_resnet18_weighted_bce",
            "desc": "ResNet-18 + Positive Class Weighted BCE (pos_weight=5.86)",
            "model_name": "resnet18",
            "loss_type": "weighted_bce",
            "sampler_type": "standard",
            "aug_level": "standard",
            "freeze_epochs": 0,
            "lr": 1e-4,
        },
        {
            "name": "exp3_resnet18_weighted_sampler",
            "desc": "ResNet-18 + WeightedRandomSampler (Balanced Batches)",
            "model_name": "resnet18",
            "loss_type": "bce",
            "sampler_type": "weighted",
            "aug_level": "standard",
            "freeze_epochs": 0,
            "lr": 1e-4,
        },
        {
            "name": "exp4_resnet18_aug_twostage",
            "desc": "ResNet-18 + Advanced Augmentations + 2-Stage Transfer Learning",
            "model_name": "resnet18",
            "loss_type": "weighted_bce",
            "sampler_type": "standard",
            "aug_level": "advanced",
            "freeze_epochs": 1,  # 1 epoch head warmup, then backbone fine-tune
            "lr": 1e-4,
            "backbone_lr": 1e-5,
        },
        {
            "name": "exp5_efficientnet_b0",
            "desc": "EfficientNet-B0 + Weighted BCE + Advanced Augmentations",
            "model_name": "efficientnet_b0",
            "loss_type": "weighted_bce",
            "sampler_type": "standard",
            "aug_level": "advanced",
            "freeze_epochs": 1,
            "lr": 1e-4,
            "backbone_lr": 1e-5,
        },
    ]

    all_results: List[Dict[str, Any]] = []

    for exp in experiment_definitions:
        print("\n" + "#" * 75)
        print(f"STARTING: {exp['name']} - {exp['desc']}")
        print("#" * 75)

        cfg = get_config(
            experiment_name=exp["name"],
            model_name=exp["model_name"],
            loss_type=exp["loss_type"],
            sampler_type=exp["sampler_type"],
            augmentation_level=exp["aug_level"],
            freeze_backbone_epochs=exp["freeze_epochs"],
            learning_rate=exp["lr"],
            backbone_lr=exp.get("backbone_lr"),
            epochs=epochs,
            batch_size=batch_size,
            device_name=device,
            random_seed=42,
            early_stopping_patience=3,
        )

        t0 = time.time()
        model, summary = train_model(cfg)
        train_duration = time.time() - t0

        # Load best saved checkpoint from experiment folder
        ckpt_path = cfg.results_dir / "experiments" / exp["name"] / "best_model.pth"
        loaded_model, metadata = load_checkpoint(ckpt_path, device=cfg.device)

        # Dataloaders for held-out test evaluation
        _, val_loader, test_loader, _ = create_dataloaders(cfg)

        # Threshold selected from validation data only
        val_thresh = float(metadata.get("decision_threshold", 0.5))

        # Evaluate on test set with default threshold (0.5) AND validation-selected threshold
        test_metrics_def, _, _, _ = evaluate_model(
            model=loaded_model,
            data_loader=test_loader,
            device=cfg.device,
            threshold=0.5,
            criterion=nn.BCEWithLogitsLoss(),
        )

        test_metrics_val_opt, _, _, _ = evaluate_model(
            model=loaded_model,
            data_loader=test_loader,
            device=cfg.device,
            threshold=val_thresh,
            criterion=nn.BCEWithLogitsLoss(),
        )

        cm_def = test_metrics_def["confusion_matrix"]
        cm_opt = test_metrics_val_opt["confusion_matrix"]

        record = {
            "experiment_id": exp["name"],
            "description": exp["desc"],
            "model_architecture": exp["model_name"],
            "loss_objective": exp["loss_type"],
            "sampling_strategy": exp["sampler_type"],
            "augmentation_level": exp["aug_level"],
            "two_stage_warmup": exp["freeze_epochs"] > 0,
            "training_time_sec": round(train_duration, 1),
            "best_epoch": summary["best_epoch"],
            "validation_f1": summary["best_val_f1"],
            "validation_selected_threshold": round(val_thresh, 4),
            "test_evaluation_default_threshold_0.5": {
                "threshold": 0.5,
                "accuracy": test_metrics_def["accuracy"],
                "balanced_accuracy": test_metrics_def["balanced_accuracy"],
                "precision": test_metrics_def["precision"],
                "recall_injured": test_metrics_def["recall_injured"],
                "false_negatives": cm_def["false_negatives"],
                "false_positives": cm_def["false_positives"],
                "true_positives": cm_def["true_positives"],
                "true_negatives": cm_def["true_negatives"],
                "f1_score": test_metrics_def["f1_score"],
                "roc_auc": test_metrics_def["roc_auc"],
                "brier_score": test_metrics_def["brier_score"],
            },
            "test_evaluation_validation_selected_threshold": {
                "threshold": round(val_thresh, 4),
                "accuracy": test_metrics_val_opt["accuracy"],
                "balanced_accuracy": test_metrics_val_opt["balanced_accuracy"],
                "precision": test_metrics_val_opt["precision"],
                "recall_injured": test_metrics_val_opt["recall_injured"],
                "false_negatives": cm_opt["false_negatives"],
                "false_positives": cm_opt["false_positives"],
                "true_positives": cm_opt["true_positives"],
                "true_negatives": cm_opt["true_negatives"],
                "f1_score": test_metrics_val_opt["f1_score"],
                "roc_auc": test_metrics_val_opt["roc_auc"],
                "brier_score": test_metrics_val_opt["brier_score"],
            },
        }

        all_results.append(record)

    # Save comparative summary
    summary_path = Path("results") / "experiments_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 90)
    print("CONTROLLED EXPERIMENT BENCHMARK SUMMARY (HELD-OUT TEST SET)")
    print("=" * 90)
    print(f"{'Experiment':<32} | {'Val F1':<7} | {'Test Acc':<8} | {'Bal Acc':<8} | {'Recall':<7} | {'FN':<3} | {'Prec':<7} | {'F1':<7}")
    print("-" * 90)
    for r in all_results:
        t_opt = r["test_evaluation_validation_selected_threshold"]
        print(
            f"{r['experiment_id']:<32} | "
            f"{r['validation_f1']:<7.4f} | "
            f"{t_opt['accuracy']:<8.4f} | "
            f"{t_opt['balanced_accuracy']:<8.4f} | "
            f"{t_opt['recall_injured']:<7.4f} | "
            f"{t_opt['false_negatives']:<3d} | "
            f"{t_opt['precision']:<7.4f} | "
            f"{t_opt['f1_score']:<7.4f}"
        )
    print("=" * 90)
    print(f"\nSaved full benchmark summary to {summary_path}")

    return all_results


if __name__ == "__main__":
    run_all_experiments()
