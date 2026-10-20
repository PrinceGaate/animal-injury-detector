"""Model architecture definitions, transfer learning utilities, and checkpoint management.

Supports ResNet-18, EfficientNet-B0, and MobileNetV3-Small with unified single-logit
classification heads, backbone freezing/unfreezing, and self-describing checkpoints.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision import models

from .config import CLASS_TO_IDX, Config, IDX_TO_CLASS, get_device

logger = logging.getLogger(__name__)


def build_model(
    model_name: str = "resnet18",
    pretrained: bool = True,
    dropout_rate: float = 0.2,
) -> nn.Module:
    """Instantiate a computer vision model with a single-logit binary classification head.

    Args:
        model_name: One of ['resnet18', 'efficientnet_b0', 'mobilenet_v3_small'].
        pretrained: If True, loads official ImageNet pre-trained weights.
        dropout_rate: Dropout probability before the linear projection layer.

    Returns:
        nn.Module: PyTorch model outputting 1 raw logit per input sample.
    """
    clean_name = model_name.strip().lower()

    if clean_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        if dropout_rate > 0.0:
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 1),
            )
        else:
            model.fc = nn.Linear(in_features, 1)

    elif clean_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 1),
        )

    elif clean_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[0].in_features  # 1024
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 1),
        )

    else:
        raise ValueError(
            f"Unsupported model_name '{model_name}'. "
            "Choose from ['resnet18', 'efficientnet_b0', 'mobilenet_v3_small']."
        )

    # Attach attribute for easy introspection
    model.model_architecture_name = clean_name
    return model


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """Freeze or unfreeze the backbone feature extractor, keeping the head trainable."""
    arch = getattr(model, "model_architecture_name", "resnet18")

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = not freeze

    # Unfreeze the classification head always
    head: nn.Module
    if hasattr(model, "fc"):
        head = model.fc
    elif hasattr(model, "classifier"):
        head = model.classifier
    else:
        raise AttributeError(f"Could not identify classification head for {arch}")

    for param in head.parameters():
        param.requires_grad = True

    status = "Frozen" if freeze else "Unfrozen"
    logger.info("Backbone %s for %s. Trainable parameters: %d", status, arch, count_trainable_parameters(model))


def count_trainable_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    config: Config,
    save_path: Union[str, Path],
    epoch: int = 0,
    val_metrics: Optional[Dict[str, float]] = None,
    decision_threshold: float = 0.5,
) -> Path:
    """Save model state along with preprocessing, architecture, and calibration metadata."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    arch_name = getattr(model, "model_architecture_name", config.model_name)

    payload: Dict[str, Any] = {
        "model_state": model_state,
        "architecture": arch_name,
        "num_classes": 1,
        "decision_threshold": float(decision_threshold),
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
        "preprocessing": {
            "image_size": config.image_size,
            "mean": list(config.norm_mean),
            "std": list(config.norm_std),
        },
        "config": config.to_dict(),
        "epoch": epoch,
        "val_metrics": val_metrics or {},
    }

    torch.save(payload, path)
    logger.info("Saved checkpoint [%s] to %s", arch_name, path)
    return path


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load model and metadata from a saved checkpoint file.

    Inspects metadata to instantiate the exact matching architecture (ResNet-18, EfficientNet, etc.).
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {path}\n"
            "Please train the model first with train.py or supply a valid path via --checkpoint."
        )

    target_device = device or get_device()
    checkpoint = torch.load(path, map_location=target_device)

    # Detect whether checkpoint is autoencoder or classifier
    if isinstance(checkpoint, dict) and "encoder_conv.0.weight" in checkpoint:
        raise ValueError(
            f"Checkpoint '{path}' appears to be an unsupervised ResNet-18 Autoencoder, "
            "not a binary classifier. Autoencoder weights cannot be loaded into the binary classifier."
        )

    metadata: Dict[str, Any] = {}
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        metadata = {k: v for k, v in checkpoint.items() if k != "model_state"}
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(f"Unrecognized checkpoint format in {path}: {type(checkpoint)}")

    # Detect architecture from metadata or infer from weight keys
    arch = metadata.get("architecture", "resnet18")
    if arch == "resnet18_binary":
        arch = "resnet18"

    if "features.0.0.weight" in state_dict:
        arch = "efficientnet_b0"
    elif "features.0.0.weight" not in state_dict and any(k.startswith("features.") for k in state_dict):
        arch = "mobilenet_v3_small"

    has_dropout_seq = any(k.startswith("fc.1.") or k.startswith("classifier.1.") for k in state_dict.keys())
    model = build_model(model_name=arch, pretrained=False, dropout_rate=0.2 if has_dropout_seq else 0.0)

    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load checkpoint into {arch} binary classifier: {e}\n"
            "Verify that the checkpoint architecture matches."
        ) from e

    model.to(target_device)
    model.eval()

    return model, metadata
