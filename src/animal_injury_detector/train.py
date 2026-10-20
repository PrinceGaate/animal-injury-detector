"""Training pipeline with transfer learning, class weighting, focal loss, and threshold tuning.

Supports ResNet-18, EfficientNet-B0, MobileNetV3-Small, two-stage fine-tuning,
weighted random sampling, and validation-driven threshold optimization.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .config import Config, get_config, seed_everything
from .dataset import create_dataloaders
from .evaluate import evaluate_model, find_optimal_threshold
from .model import build_model, count_trainable_parameters, freeze_backbone, save_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class BinaryFocalLossWithLogits(nn.Module):
    """Binary Focal Loss with numerical stability from raw logits."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal_weight = alpha_t * ((1.0 - p_t) ** self.gamma)
        loss = focal_weight * bce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss_criterion(config: Config, pos_weight_val: float, device: torch.device) -> nn.Module:
    """Build the specified loss function."""
    loss_type = config.loss_type.lower()
    if loss_type == "focal":
        logger.info("Using BinaryFocalLossWithLogits (alpha=0.25, gamma=2.0)")
        return BinaryFocalLossWithLogits(alpha=0.25, gamma=2.0)
    elif loss_type == "bce":
        logger.info("Using unweighted BCEWithLogitsLoss")
        return nn.BCEWithLogitsLoss()
    else:  # "weighted_bce"
        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
        logger.info("Using BCEWithLogitsLoss with positive class weight: %.3f", pos_weight_val)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    """Run one epoch of training over the training DataLoader."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Train]", leave=False)
    for inputs, targets, _ in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return epoch_loss


def build_optimizer_and_scheduler(
    model: nn.Module,
    config: Config,
    backbone_frozen: bool,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """Build optimizer with differential learning rates if backbone is unfrozen."""
    if backbone_frozen:
        # Only head parameters
        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(head_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        # Differential LR: backbone at lower rate, head at base rate
        head_params = []
        backbone_params = []
        arch = getattr(model, "model_architecture_name", "resnet18")

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "fc" in name or "classifier" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        bb_lr = config.backbone_lr if config.backbone_lr is not None else config.learning_rate * 0.1
        param_groups = [
            {"params": backbone_params, "lr": bb_lr},
            {"params": head_params, "lr": config.learning_rate},
        ]
        optimizer = AdamW(param_groups, weight_decay=config.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    return optimizer, scheduler


def train_model(
    config: Config,
    custom_samples: Optional[Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], Optional[List[Tuple[Path, int]]]]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Execute complete training and validation pipeline."""
    seed_everything(config.random_seed)
    device = config.device
    logger.info("Initializing experiment '%s' on device: %s", config.experiment_name, device)

    # Output directories
    exp_dir = config.results_dir / "experiments" / config.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Dataloaders
    train_loader, val_loader, test_loader, pos_weight_val = create_dataloaders(
        config, custom_samples=custom_samples
    )

    # Loss criterion
    criterion = get_loss_criterion(config, pos_weight_val, device)

    # Model
    model = build_model(
        model_name=config.model_name,
        pretrained=config.pretrained,
        dropout_rate=config.dropout_rate,
    )
    model.to(device)

    # Initial freezing state
    backbone_frozen = config.freeze_backbone_epochs > 0
    if backbone_frozen:
        freeze_backbone(model, freeze=True)

    optimizer, scheduler = build_optimizer_and_scheduler(model, config, backbone_frozen=backbone_frozen)

    best_val_f1 = -1.0
    best_epoch = -1
    best_optimal_threshold = config.decision_threshold
    patience_counter = 0

    history: Dict[str, List[Any]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_balanced_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_false_negatives": [],
        "optimal_threshold": [],
    }

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {config.experiment_name.upper()}")
    print(f"Architecture: {config.model_name} | Pretrained: {config.pretrained}")
    print(f"Loss: {config.loss_type} | Sampler: {config.sampler_type} | Aug: {config.augmentation_level}")
    print(f"Epochs: {config.epochs} | Batch: {config.batch_size} | LR: {config.learning_rate}")
    print(f"Freeze Warmup: {config.freeze_backbone_epochs} epochs | Device: {device}")
    print("=" * 70)

    for epoch in range(1, config.epochs + 1):
        # Handle unfreezing backbone after warmup
        if backbone_frozen and epoch > config.freeze_backbone_epochs:
            logger.info("Unfreezing backbone at epoch %d for end-to-end fine-tuning.", epoch)
            freeze_backbone(model, freeze=False)
            backbone_frozen = False
            optimizer, scheduler = build_optimizer_and_scheduler(model, config, backbone_frozen=False)

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=config.epochs,
        )

        # Validation evaluation using deterministic decision threshold
        val_metrics, y_val_true, y_val_probs, _ = evaluate_model(
            model=model,
            data_loader=val_loader,
            device=device,
            threshold=config.decision_threshold,
            criterion=criterion,
        )

        # Optional threshold search on validation predictions
        opt_thresh, opt_score = find_optimal_threshold(y_val_true, y_val_probs, metric="f1")

        val_loss = val_metrics.get("loss", 0.0)
        val_acc = val_metrics["accuracy"]
        val_bal = val_metrics["balanced_accuracy"]
        val_prec = val_metrics["precision"]
        val_rec = val_metrics["recall_injured"]
        val_f1 = val_metrics["f1_score"]
        val_fn = val_metrics["confusion_matrix"]["false_negatives"]

        scheduler.step(val_f1)

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_accuracy"].append(val_acc)
        history["val_balanced_acc"].append(val_bal)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_rec)
        history["val_f1"].append(val_f1)
        history["val_false_negatives"].append(val_fn)
        history["optimal_threshold"].append(round(opt_thresh, 4))

        print(
            f"Epoch {epoch:02d}/{config.epochs:02d} | "
            f"TrLoss: {train_loss:.4f} | "
            f"ValLoss: {val_loss:.4f} | "
            f"BalAcc: {val_bal:.4f} | "
            f"F1: {val_f1:.4f} (Opt: {opt_score:.4f} @ {opt_thresh:.2f}) | "
            f"Rec: {val_rec:.4f} | "
            f"FN: {val_fn}"
        )

        # Selection criterion: Best Validation F1
        score_to_compare = opt_score if config.optimize_threshold_val else val_f1
        if score_to_compare > best_val_f1:
            best_val_f1 = score_to_compare
            best_epoch = epoch
            best_optimal_threshold = opt_thresh if config.optimize_threshold_val else config.decision_threshold
            patience_counter = 0

            # Save to standard checkpoint path and experiment checkpoint path
            save_checkpoint(
                model=model,
                config=config,
                save_path=config.best_checkpoint_path,
                epoch=epoch,
                val_metrics=val_metrics,
                decision_threshold=best_optimal_threshold,
            )
            save_checkpoint(
                model=model,
                config=config,
                save_path=exp_dir / "best_model.pth",
                epoch=epoch,
                val_metrics=val_metrics,
                decision_threshold=best_optimal_threshold,
            )
            print(f"  >>> Checkpoint saved: New best validation score = {best_val_f1:.4f} (Threshold: {best_optimal_threshold:.2f})")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"\n[Early Stopping] No improvement for {config.early_stopping_patience} epochs.")
                break

    # Save training history and summary
    summary_data = {
        "experiment_name": config.experiment_name,
        "model_name": config.model_name,
        "best_epoch": best_epoch,
        "best_val_f1": round(best_val_f1, 4),
        "selected_threshold": round(best_optimal_threshold, 4),
        "history": history,
        "config": config.to_dict(),
    }

    with open(exp_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    with open(config.results_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSaved training history to {exp_dir / 'training_summary.json'}")
    return model, summary_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Animal Injury Detector with controlled experiments.")
    parser.add_argument("--experiment-name", type=str, default="baseline_resnet18", help="Experiment identifier.")
    parser.add_argument("--model-name", type=str, default="resnet18", choices=["resnet18", "efficientnet_b0", "mobilenet_v3_small"], help="Model architecture.")
    parser.add_argument("--loss-type", type=str, default="weighted_bce", choices=["weighted_bce", "bce", "focal"], help="Loss objective.")
    parser.add_argument("--sampler-type", type=str, default="standard", choices=["standard", "weighted"], help="Batch sampling strategy.")
    parser.add_argument("--aug-level", type=str, default="standard", choices=["none", "standard", "advanced"], help="Data augmentation level.")
    parser.add_argument("--freeze-epochs", type=int, default=0, help="Number of warmup epochs with frozen backbone.")
    parser.add_argument("--epochs", type=int, default=10, help="Maximum epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--backbone-lr", type=float, default=None, help="Backbone learning rate.")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cpu', 'cuda', 'mps').")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience.")
    args = parser.parse_args()

    config = get_config(
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        loss_type=args.loss_type,
        sampler_type=args.sampler_type,
        augmentation_level=args.aug_level,
        freeze_backbone_epochs=args.freeze_epochs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        backbone_lr=args.backbone_lr,
        device_name=args.device,
        random_seed=args.seed,
        early_stopping_patience=args.patience,
    )

    train_model(config)


if __name__ == "__main__":
    main()
