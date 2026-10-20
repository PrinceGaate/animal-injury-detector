"""Configuration module for Animal Injury Detector.

Centralizes hyperparameters, dataset paths, class mappings, preprocessing constants,
transfer learning controls, and device selection across training, evaluation, and inference.
"""

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# Canonical class mappings
CLASS_TO_IDX: Dict[str, int] = {
    "healthy": 0,
    "injured": 1,
}

IDX_TO_CLASS: Dict[int, str] = {
    0: "healthy",
    1: "injured",
}

# ImageNet default normalization statistics
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def get_default_data_dir() -> Path:
    """Find the default dataset directory based on standard repository locations."""
    env_dir = os.environ.get("ANIMAL_INJURY_DATA_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)

    repo_root = Path(__file__).resolve().parents[2]
    candidate_paths = [
        repo_root / "dataset" / "injuries",
        repo_root / "Major" / "injuries",
        repo_root / "injuries",
    ]

    for p in candidate_paths:
        if p.exists() and (p / "train").exists():
            return p

    # Default fallback
    return repo_root / "dataset" / "injuries"


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Select appropriate compute device portably across CUDA, MPS, and CPU."""
    if preferred:
        pref = preferred.lower()
        if pref.startswith("cuda") and torch.cuda.is_available():
            return torch.device(pref)
        if pref == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if pref == "cpu":
            return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class Config:
    """Configuration dataclass holding runtime hyperparameters and experiment settings."""

    # Paths & Experiment Naming
    experiment_name: str = "baseline_resnet18"
    data_dir: Path = field(default_factory=get_default_data_dir)
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    best_checkpoint_name: str = "best_model.pth"

    # Preprocessing
    image_size: int = 224
    norm_mean: Tuple[float, float, float] = IMAGENET_MEAN
    norm_std: Tuple[float, float, float] = IMAGENET_STD
    augmentation_level: str = "standard"  # "none", "standard", "advanced"

    # Training Hyperparameters
    batch_size: int = 32
    num_workers: int = 0  # 0 is safe on Windows to avoid multiprocessing issues
    learning_rate: float = 1e-4
    backbone_lr: Optional[float] = None  # Differential LR for backbone fine-tuning
    weight_decay: float = 1e-4
    epochs: int = 10
    early_stopping_patience: int = 3
    random_seed: int = 42

    # Transfer Learning & Fine-Tuning
    freeze_backbone_epochs: int = 0  # Freeze backbone for initial epochs then unfreeze

    # Imbalance & Sampling Strategy
    loss_type: str = "weighted_bce"  # "bce", "weighted_bce", "focal"
    sampler_type: str = "standard"   # "standard", "weighted"
    use_pos_weight: bool = True      # Used when loss_type == "weighted_bce"
    decision_threshold: float = 0.5
    optimize_threshold_val: bool = True  # Automatically tune threshold on validation set

    # Architecture
    model_name: str = "resnet18"  # "resnet18", "efficientnet_b0", "mobilenet_v3_small"
    pretrained: bool = True
    dropout_rate: float = 0.2

    # Device
    device_name: Optional[str] = None

    @property
    def device(self) -> torch.device:
        return get_device(self.device_name)

    @property
    def best_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.best_checkpoint_name

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a JSON-compatible dictionary with portable relative paths."""
        d = asdict(self)
        repo_root = Path(__file__).resolve().parents[2]
        for key in ["data_dir", "checkpoint_dir", "results_dir"]:
            val = getattr(self, key)
            try:
                d[key] = str(Path(val).resolve().relative_to(repo_root))
            except (ValueError, AttributeError):
                d[key] = str(val)
        d["device"] = str(self.device)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Instantiate Config from a dictionary, converting path strings back to Path."""
        clean_data = dict(data)
        if "data_dir" in clean_data:
            clean_data["data_dir"] = Path(clean_data["data_dir"])
        if "checkpoint_dir" in clean_data:
            clean_data["checkpoint_dir"] = Path(clean_data["checkpoint_dir"])
        if "results_dir" in clean_data:
            clean_data["results_dir"] = Path(clean_data["results_dir"])
        if "device" in clean_data and "device_name" not in clean_data:
            clean_data["device_name"] = str(clean_data.pop("device"))
        # Drop unrecognized keys to ensure backwards compatibility
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in clean_data.items() if k in valid_keys}
        return cls(**filtered)


def get_config(**kwargs: Any) -> Config:
    """Factory helper to obtain a Config instance with custom overrides."""
    return Config(**kwargs)
