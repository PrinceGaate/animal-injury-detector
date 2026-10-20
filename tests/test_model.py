"""Tests for model architectures, checkpoint serialization, threshold optimization, and inference."""

from pathlib import Path
import pytest
from PIL import Image
import torch
import torch.nn as nn

from animal_injury_detector.config import Config, get_config
from animal_injury_detector.evaluate import compute_binary_metrics, find_optimal_threshold
from animal_injury_detector.infer import AnimalInjuryPredictor
from animal_injury_detector.model import build_model, freeze_backbone, load_checkpoint, save_checkpoint


def test_model_architectures_forward_shape():
    """Verify that all supported architectures output a single logit (B, 1)."""
    for arch in ["resnet18", "efficientnet_b0", "mobilenet_v3_small"]:
        model = build_model(model_name=arch, pretrained=False)
        model.eval()

        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (2, 1)
        assert output.dtype == torch.float32


def test_backbone_freezing():
    """Verify that freeze_backbone sets requires_grad appropriately."""
    model = build_model(model_name="resnet18", pretrained=False)
    freeze_backbone(model, freeze=True)

    # Backbone parameters should be frozen
    assert not model.conv1.weight.requires_grad
    # Head parameters should remain trainable
    for p in model.fc.parameters():
        assert p.requires_grad

    freeze_backbone(model, freeze=False)
    assert model.conv1.weight.requires_grad


def test_checkpoint_roundtrip_save_and_load(tmp_path: Path):
    """Verify checkpoint saving with metadata and successful loading."""
    config = get_config(checkpoint_dir=tmp_path)
    model = build_model(model_name="resnet18", pretrained=False)

    ckpt_path = tmp_path / "test_model.pth"
    save_checkpoint(
        model=model,
        config=config,
        save_path=ckpt_path,
        epoch=1,
        val_metrics={"f1_score": 0.85},
        decision_threshold=0.42,
    )

    assert ckpt_path.exists()

    loaded_model, metadata = load_checkpoint(ckpt_path)
    assert isinstance(loaded_model, nn.Module)
    assert metadata["architecture"] == "resnet18"
    assert metadata["epoch"] == 1
    assert metadata["decision_threshold"] == 0.42
    assert metadata["val_metrics"]["f1_score"] == 0.85
    assert "preprocessing" in metadata
    assert metadata["preprocessing"]["image_size"] == 224


def test_incompatible_autoencoder_rejection(tmp_path: Path):
    """Verify that attempting to load an autoencoder checkpoint raises an informative error."""
    autoencoder_mock = {
        "encoder_conv.0.weight": torch.randn(64, 3, 7, 7),
        "decoder.0.weight": torch.randn(256, 512, 3, 3),
    }
    ae_path = tmp_path / "fake_autoencoder.pth"
    torch.save(autoencoder_mock, ae_path)

    with pytest.raises(ValueError, match="Autoencoder"):
        load_checkpoint(ae_path)


def test_missing_checkpoint_error(tmp_path: Path):
    """Verify clean error on non-existent checkpoint path."""
    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        load_checkpoint(tmp_path / "does_not_exist.pth")


def test_inference_predictor_pipeline(tmp_path: Path):
    """Verify AnimalInjuryPredictor works end-to-end on a synthetic image."""
    config = get_config(checkpoint_dir=tmp_path)
    model = build_model(pretrained=False)
    ckpt_path = tmp_path / "test_ckpt.pth"
    save_checkpoint(model, config, ckpt_path, decision_threshold=0.35)

    # Synthetic test image
    test_img_path = tmp_path / "sample_dog.jpg"
    Image.new("RGB", (300, 300), color=(120, 100, 80)).save(test_img_path)

    predictor = AnimalInjuryPredictor(checkpoint_path=ckpt_path)
    res = predictor.predict_image(test_img_path)

    assert "estimated_injury_score" in res
    assert "probability_injured" in res
    assert 0.0 <= res["probability_injured"] <= 1.0
    assert res["predicted_label"] in ("injured", "healthy")
    assert isinstance(res["is_injured"], bool)
    assert res["decision_threshold"] == 0.35
    assert res["image_path"] == str(test_img_path)


def test_binary_metrics_calculation():
    """Verify binary metrics computation including precision, recall, and false negatives."""
    import numpy as np

    # 4 samples: 2 healthy (0), 2 injured (1)
    y_true = np.array([0, 0, 1, 1])
    # Predictions: 1st correct, 2nd FP, 3rd FN, 4th TP
    y_probs = np.array([0.1, 0.9, 0.3, 0.8])

    metrics = compute_binary_metrics(y_true, y_probs, threshold=0.5)

    assert metrics["total_samples"] == 4
    assert metrics["confusion_matrix"]["true_positives"] == 1
    assert metrics["confusion_matrix"]["false_negatives"] == 1
    assert metrics["confusion_matrix"]["false_positives"] == 1
    assert metrics["confusion_matrix"]["true_negatives"] == 1
    assert metrics["recall_injured"] == 0.5
    assert metrics["false_negatives"] == 1
    assert metrics["balanced_accuracy"] == 0.5


def test_threshold_optimization():
    """Verify optimal threshold search on validation predictions."""
    import numpy as np

    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.35, 0.4, 0.8, 0.9])

    # At threshold 0.38, all 3 positives >= 0.38 and negatives < 0.38 -> F1 = 1.0
    best_t, best_score = find_optimal_threshold(y_true, y_probs, metric="f1")
    assert best_score == 1.0
    assert 0.35 < best_t <= 0.40
