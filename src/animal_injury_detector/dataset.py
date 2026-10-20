"""Dataset loading, class mapping, augmentations, and split validation.

Supports standard/advanced augmentations, weighted random sampling for class balance,
and prefix-grouped leakage-resistant splitting.
"""

from collections import defaultdict
import logging
from pathlib import Path
import random
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from .config import Config, IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def map_class_name_to_label(folder_name: str) -> int:
    """Map class directory name to binary label.

    Healthy -> 0 (negative)
    Injured/Injuries -> 1 (positive)
    """
    clean = folder_name.strip().lower()
    if clean in {"healthy", "normal", "non_injured", "non-injured", "uninjured"}:
        return 0
    if clean in {"injuries", "injury", "injured", "injured animal", "wound", "wounds"}:
        return 1
    raise ValueError(
        f"Unrecognized class folder name '{folder_name}'. "
        "Expected one of ['healthy', 'injuries', 'injured']."
    )


def extract_base_prefix(filename: str) -> str:
    """Extract source image identifier before Roboflow hash/extension."""
    for pattern in ["_jpg.rf.", "_png.rf.", "_jpeg.rf."]:
        if pattern in filename:
            return filename.split(pattern)[0]
    return Path(filename).stem


def get_train_transforms(
    image_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
    level: str = "standard",
) -> transforms.Compose:
    """Compose data augmentation transforms for training based on selected level."""
    if level == "none":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ])

    if level == "advanced":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08), value="random"),
        ])

    # Standard baseline augmentations
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])


def get_eval_transforms(
    image_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> transforms.Compose:
    """Compose deterministic transforms for validation, testing, and inference."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])


class AnimalInjuryDataset(Dataset):
    """PyTorch Dataset for binary animal injury classification."""

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        """Initialize with a list of (image_path, label) pairs."""
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_path, label = self.samples[idx]
        try:
            with Image.open(img_path) as raw_img:
                img = raw_img.convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise RuntimeError(f"Corrupt or unreadable image file: {img_path}") from e

        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return tensor, label_tensor, str(img_path)


def scan_split_directory(split_dir: Path) -> List[Tuple[Path, int]]:
    """Scan a split directory (e.g. train/) and collect (path, label) pairs."""
    if not split_dir.exists() or not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    samples: List[Tuple[Path, int]] = []
    class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]

    if not class_dirs:
        raise ValueError(
            f"No class subdirectories found in {split_dir}. "
            "Expected subfolders like 'Healthy' and 'injuries'."
        )

    for class_dir in class_dirs:
        try:
            label = map_class_name_to_label(class_dir.name)
        except ValueError:
            logger.warning("Skipping unknown subdirectory: %s", class_dir)
            continue

        for file_path in class_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                samples.append((file_path, label))

    if not samples:
        raise ValueError(f"No valid image files found in {split_dir}.")

    return samples


def validate_dataset_structure(data_dir: Path) -> Dict[str, Dict[str, int]]:
    """Verify presence of splits and report sample distribution per class."""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            "Please check config.py or pass --data-dir pointing to your dataset root."
        )

    stats: Dict[str, Dict[str, int]] = {}
    for split in ["train", "valid", "test"]:
        split_path = data_dir / split
        if not split_path.exists():
            continue
        samples = scan_split_directory(split_path)
        num_healthy = sum(1 for _, lbl in samples if lbl == 0)
        num_injured = sum(1 for _, lbl in samples if lbl == 1)
        stats[split] = {
            "healthy": num_healthy,
            "injured": num_injured,
            "total": len(samples),
        }

    if "train" not in stats or stats["train"]["total"] == 0:
        raise ValueError(f"No training data found in {data_dir / 'train'}.")
    if "valid" not in stats or stats["valid"]["total"] == 0:
        raise ValueError(f"No validation data found in {data_dir / 'valid'}.")

    return stats


def compute_positive_weight(train_samples: List[Tuple[Path, int]]) -> float:
    """Compute pos_weight = num_negative / num_positive for BCEWithLogitsLoss."""
    num_neg = sum(1 for _, lbl in train_samples if lbl == 0)
    num_pos = sum(1 for _, lbl in train_samples if lbl == 1)

    if num_pos == 0:
        raise ValueError("Training split contains 0 positive (injured) samples!")
    if num_neg == 0:
        raise ValueError("Training split contains 0 negative (healthy) samples!")

    pos_weight = float(num_neg / num_pos)
    return pos_weight


def create_weighted_sampler(train_samples: List[Tuple[Path, int]]) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler to draw balanced mini-batches."""
    labels = [lbl for _, lbl in train_samples]
    num_neg = sum(1 for l in labels if l == 0)
    num_pos = sum(1 for l in labels if l == 1)

    weight_per_class = {
        0: 1.0 / num_neg if num_neg > 0 else 0.0,
        1: 1.0 / num_pos if num_pos > 0 else 0.0,
    }

    sample_weights = [weight_per_class[l] for l in labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def create_leakage_resistant_grouped_splits(
    all_samples: List[Tuple[Path, int]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """Group samples by base subject prefix so augmented versions never cross splits.

    Strictly guarantees zero subject prefix overlap across train, validation, and test sets.
    """
    rng = random.Random(seed)

    # Group files by base_prefix
    prefix_to_samples: Dict[str, List[Tuple[Path, int]]] = defaultdict(list)
    prefix_to_labels: Dict[str, Set[int]] = defaultdict(set)
    for path, label in all_samples:
        prefix = extract_base_prefix(path.name)
        prefix_to_samples[prefix].append((path, label))
        prefix_to_labels[prefix].add(label)

    # Any prefix with mixed labels across classes (e.g. generic 'images' scraped files)
    # is assigned exclusively to training to prevent string collision across evaluation splits.
    mixed_prefixes = sorted([p for p, lbls in prefix_to_labels.items() if len(lbls) > 1])
    clean_healthy = sorted([p for p, lbls in prefix_to_labels.items() if lbls == {0}])
    clean_injured = sorted([p for p, lbls in prefix_to_labels.items() if lbls == {1}])

    rng.shuffle(clean_healthy)
    rng.shuffle(clean_injured)

    def split_groups(prefixes: List[str]) -> Tuple[List[str], List[str], List[str]]:
        n_total = len(prefixes)
        n_test = max(1, int(n_total * test_ratio))
        n_val = max(1, int(n_total * val_ratio))
        test_p = prefixes[:n_test]
        val_p = prefixes[n_test : n_test + n_val]
        train_p = prefixes[n_test + n_val :]
        return train_p, val_p, test_p

    h_tr, h_val, h_test = split_groups(clean_healthy)
    i_tr, i_val, i_test = split_groups(clean_injured)

    train_prefixes = h_tr + i_tr + mixed_prefixes
    val_prefixes = h_val + i_val
    test_prefixes = h_test + i_test

    train_samples = [s for p in train_prefixes for s in prefix_to_samples[p]]
    val_samples = [s for p in val_prefixes for s in prefix_to_samples[p]]
    test_samples = [s for p in test_prefixes for s in prefix_to_samples[p]]

    return train_samples, val_samples, test_samples


def create_dataloaders(
    config: Config,
    custom_samples: Optional[Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], Optional[List[Tuple[Path, int]]]]] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], float]:
    """Create train, validation, and test dataloaders from configuration."""
    if custom_samples is not None:
        train_samples, val_samples, test_samples = custom_samples
    else:
        data_dir = Path(config.data_dir)
        validate_dataset_structure(data_dir)

        train_samples = scan_split_directory(data_dir / "train")
        val_samples = scan_split_directory(data_dir / "valid")

        test_samples = None
        if (data_dir / "test").exists():
            try:
                test_samples = scan_split_directory(data_dir / "test")
            except ValueError as e:
                logger.warning("Could not load test split: %s", e)

    # Class weighting
    pos_weight = compute_positive_weight(train_samples)

    # Warnings for small splits
    for split_name, split_samples in [("valid", val_samples), ("test", test_samples or [])]:
        if not split_samples:
            continue
        n_pos = sum(1 for _, lbl in split_samples if lbl == 1)
        if n_pos < 15:
            logger.warning(
                "WARNING: '%s' split has only %d positive (injured) samples. "
                "Evaluation metrics (especially recall) carry wide statistical uncertainty.",
                split_name,
                n_pos,
            )

    train_tf = get_train_transforms(
        image_size=config.image_size,
        mean=config.norm_mean,
        std=config.norm_std,
        level=config.augmentation_level,
    )
    eval_tf = get_eval_transforms(
        image_size=config.image_size,
        mean=config.norm_mean,
        std=config.norm_std,
    )

    train_ds = AnimalInjuryDataset(train_samples, transform=train_tf)
    val_ds = AnimalInjuryDataset(val_samples, transform=eval_tf)

    # Sampler handling
    sampler = None
    shuffle = True
    if config.sampler_type == "weighted":
        sampler = create_weighted_sampler(train_samples)
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader: Optional[DataLoader] = None
    if test_samples:
        test_ds = AnimalInjuryDataset(test_samples, transform=eval_tf)
        test_loader = DataLoader(
            test_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader, test_loader, pos_weight
