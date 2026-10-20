"""Animal Injury Detector.

A reproducible deep learning pipeline for binary classification of animal injuries
using PyTorch and pretrained ResNet architectures.
"""

__version__ = "1.0.0"

from .config import Config, get_config
from .dataset import AnimalInjuryDataset, create_dataloaders
from .model import build_model, load_checkpoint, save_checkpoint

__all__ = [
    "Config",
    "get_config",
    "AnimalInjuryDataset",
    "create_dataloaders",
    "build_model",
    "load_checkpoint",
    "save_checkpoint",
]
