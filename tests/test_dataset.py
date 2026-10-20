"""Tests for dataset loading, class mapping, augmentations, and grouping."""

from pathlib import Path
import pytest
from PIL import Image
import torch

from animal_injury_detector.dataset import (
    AnimalInjuryDataset,
    compute_positive_weight,
    create_leakage_resistant_grouped_splits,
    create_weighted_sampler,
    extract_base_prefix,
    get_eval_transforms,
    get_train_transforms,
    map_class_name_to_label,
    scan_split_directory,
    validate_dataset_structure,
)


def test_map_class_name_to_label():
    """Verify that healthy maps to 0 and injury maps to 1."""
    assert map_class_name_to_label("Healthy") == 0
    assert map_class_name_to_label("healthy") == 0
    assert map_class_name_to_label("normal") == 0

    assert map_class_name_to_label("injuries") == 1
    assert map_class_name_to_label("Injured") == 1
    assert map_class_name_to_label("injury") == 1

    with pytest.raises(ValueError, match="Unrecognized class folder"):
        map_class_name_to_label("unknown_class")


def test_extract_base_prefix():
    """Verify base prefix extraction across Roboflow pattern formats."""
    assert extract_base_prefix("dog_1_jpg.rf.abcdef123.jpg") == "dog_1"
    assert extract_base_prefix("cat_photo_png.rf.999888.png") == "cat_photo"
    assert extract_base_prefix("simple_dog.jpg") == "simple_dog"


def test_transforms_output_shape():
    """Verify that standard and advanced transforms produce normalized tensors of shape (3, 224, 224)."""
    img = Image.new("RGB", (300, 400), color=(100, 150, 200))

    for level in ["none", "standard", "advanced"]:
        train_tf = get_train_transforms(image_size=224, level=level)
        train_tensor = train_tf(img)
        assert train_tensor.shape == (3, 224, 224)
        assert train_tensor.dtype == torch.float32

    eval_tf = get_eval_transforms(image_size=224)
    eval_tensor = eval_tf(img)
    assert eval_tensor.shape == (3, 224, 224)
    assert eval_tensor.dtype == torch.float32


def test_synthetic_dataset_loading(tmp_path: Path):
    """Verify loading from a temporary synthetic folder structure."""
    split_dir = tmp_path / "train"
    healthy_dir = split_dir / "Healthy"
    injured_dir = split_dir / "injuries"
    healthy_dir.mkdir(parents=True)
    injured_dir.mkdir(parents=True)

    img_h = Image.new("RGB", (64, 64), color=(0, 255, 0))
    img_h.save(healthy_dir / "h1.jpg")
    img_h.save(healthy_dir / "h2.jpg")

    img_inj = Image.new("RGB", (64, 64), color=(255, 0, 0))
    img_inj.save(injured_dir / "i1.jpg")

    samples = scan_split_directory(split_dir)
    assert len(samples) == 3

    dataset = AnimalInjuryDataset(samples, transform=get_eval_transforms(image_size=128))
    assert len(dataset) == 3

    tensor, label, path_str = dataset[0]
    assert tensor.shape == (3, 128, 128)
    assert isinstance(label, torch.Tensor)
    assert label.item() in (0.0, 1.0)
    assert Path(path_str).exists()


def test_weighted_sampler_creation(tmp_path: Path):
    """Verify WeightedRandomSampler initialization."""
    samples = [
        (tmp_path / "h1.jpg", 0),
        (tmp_path / "h2.jpg", 0),
        (tmp_path / "h3.jpg", 0),
        (tmp_path / "i1.jpg", 1),
    ]
    sampler = create_weighted_sampler(samples)
    assert sampler.num_samples == 4
    assert len(sampler.weights) == 4


def test_leakage_resistant_grouping(tmp_path: Path):
    """Verify that augmented variants of the same subject do not cross splits."""
    samples = [
        # Subject A: 3 augmented images
        (tmp_path / "dogA_jpg.rf.1.jpg", 0),
        (tmp_path / "dogA_jpg.rf.2.jpg", 0),
        (tmp_path / "dogA_jpg.rf.3.jpg", 0),
        # Subject B: 2 augmented images
        (tmp_path / "dogB_jpg.rf.1.jpg", 0),
        (tmp_path / "dogB_jpg.rf.2.jpg", 0),
        # Subject C: 2 injured images
        (tmp_path / "injC_jpg.rf.1.jpg", 1),
        (tmp_path / "injC_jpg.rf.2.jpg", 1),
        # Subject D: 1 injured image
        (tmp_path / "injD_jpg.rf.1.jpg", 1),
    ]

    train_s, val_s, test_s = create_leakage_resistant_grouped_splits(
        samples, val_ratio=0.25, test_ratio=0.25, seed=42
    )

    tr_prefixes = {extract_base_prefix(p.name) for p, _ in train_s}
    val_prefixes = {extract_base_prefix(p.name) for p, _ in val_s}
    test_prefixes = {extract_base_prefix(p.name) for p, _ in test_s}

    # Zero prefix leakage across splits
    assert len(tr_prefixes.intersection(val_prefixes)) == 0
    assert len(tr_prefixes.intersection(test_prefixes)) == 0
    assert len(val_prefixes.intersection(test_prefixes)) == 0
