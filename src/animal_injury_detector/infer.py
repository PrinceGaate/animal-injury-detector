"""Inference module for Animal Injury Detector.

Generates predictions using torch.inference_mode() with strict preprocessing parity,
custom decision thresholds, and clinical triage disclaimers.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms

from .config import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_device,
)
from .dataset import SUPPORTED_EXTENSIONS
from .model import load_checkpoint

logger = logging.getLogger(__name__)


class AnimalInjuryPredictor:
    """Inference engine maintaining loaded model state and preprocessing transforms."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "checkpoints/best_model.pth",
        device: Optional[Union[str, torch.device]] = None,
        default_threshold: Optional[float] = None,
    ) -> None:
        """Initialize predictor by loading checkpoint weights, architecture, and calibration metadata."""
        if isinstance(device, str):
            self.device = get_device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = get_device()

        self.checkpoint_path = Path(checkpoint_path)
        self.model, self.metadata = load_checkpoint(self.checkpoint_path, device=self.device)

        # Fallback priority: explicit argument -> checkpoint calibrated threshold -> 0.5
        ckpt_thresh = self.metadata.get("decision_threshold")
        if default_threshold is not None:
            self.default_threshold = float(default_threshold)
        elif ckpt_thresh is not None:
            self.default_threshold = float(ckpt_thresh)
        else:
            self.default_threshold = 0.5

        # Extract preprocessing parameters saved during training to ensure strict parity
        preprocessing = self.metadata.get("preprocessing", {})
        self.image_size: int = preprocessing.get("image_size", 224)
        self.mean: List[float] = preprocessing.get("mean", list(IMAGENET_MEAN))
        self.std: List[float] = preprocessing.get("std", list(IMAGENET_STD))

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        self.architecture = self.metadata.get("architecture", getattr(self.model, "model_architecture_name", "resnet18"))

    def predict_image(
        self,
        image_path: Union[str, Path],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run prediction on a single image file using torch.inference_mode().

        Args:
            image_path: Path to target image.
            threshold: Decision boundary (defaults to self.default_threshold).

        Returns:
            Dict containing image path, injury model score estimate, predicted label,
            decision threshold, raw logit, and diagnostic disclaimer.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        try:
            with Image.open(path) as raw_img:
                img = raw_img.convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(f"Cannot read image file '{path}'. Corrupt or unsupported format.") from e

        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            logit_val = logits.item()
            prob = torch.sigmoid(logits).item()

        cutoff = threshold if threshold is not None else self.default_threshold
        is_injured = bool(prob >= cutoff)
        pred_label = "injured" if is_injured else "healthy"

        return {
            "image_path": str(path),
            "architecture": self.architecture,
            "estimated_injury_score": round(prob, 4),
            "probability_injured": round(prob, 4),
            "predicted_label": pred_label,
            "is_injured": is_injured,
            "decision_threshold": round(float(cutoff), 4),
            "raw_logit": round(logit_val, 4),
            "disclaimer": "Model estimation only. NOT a certified veterinary diagnostic certainty.",
        }

    def predict_directory(
        self,
        directory_path: Union[str, Path],
        threshold: Optional[float] = None,
        recursive: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run batch prediction on all valid image files in a directory."""
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        pattern = "**/*" if recursive else "*"
        candidates = [
            p for p in dir_path.glob(pattern)
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not candidates:
            logger.warning("No image files found in %s", dir_path)
            return []

        results: List[Dict[str, Any]] = []
        for img_p in sorted(candidates):
            try:
                res = self.predict_image(img_p, threshold=threshold)
                results.append(res)
            except Exception as e:
                logger.error("Error predicting image %s: %s", img_p, e)
                results.append({
                    "image_path": str(img_p),
                    "error": str(e),
                })

        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with Animal Injury Detector.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to checkpoint.")
    parser.add_argument("--image", type=str, default=None, help="Path to single image file.")
    parser.add_argument("--image-dir", type=str, default=None, help="Path to directory of images.")
    parser.add_argument("--threshold", type=float, default=None, help="Decision cutoff override.")
    parser.add_argument("--device", type=str, default=None, help="Compute device override.")
    parser.add_argument("--output", type=str, default=None, help="Path to save output JSON results.")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("You must provide either --image or --image-dir.")

    predictor = AnimalInjuryPredictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
        default_threshold=args.threshold,
    )

    if args.image:
        res = predictor.predict_image(args.image, threshold=args.threshold)
        print("\n" + "=" * 55)
        print("ANIMAL INJURY PREDICTION RESULT")
        print("=" * 55)
        print(f"Image       : {res['image_path']}")
        print(f"Model       : {res['architecture']}")
        print(f"Prediction  : {res['predicted_label'].upper()}")
        print(f"Score       : {res['estimated_injury_score']:.2%}")
        print(f"Threshold   : {res['decision_threshold']}")
        print(f"Note        : {res['disclaimer']}")
        print("=" * 55 + "\n")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"Saved result to {args.output}")

    elif args.image_dir:
        results = predictor.predict_directory(args.image_dir, threshold=args.threshold)
        print(f"\nProcessed {len(results)} images in {args.image_dir}:")
        injured_count = sum(1 for r in results if r.get("is_injured"))
        healthy_count = sum(1 for r in results if not r.get("is_injured") and "error" not in r)
        error_count = sum(1 for r in results if "error" in r)

        print(f"  Injured: {injured_count} | Healthy: {healthy_count} | Errors: {error_count}\n")
        for r in results[:10]:
            if "error" in r:
                print(f"  {r['image_path']} -> ERROR: {r['error']}")
            else:
                print(f"  {r['image_path']} -> {r['predicted_label'].upper()} ({r['estimated_injury_score']:.1%})")
        if len(results) > 10:
            print(f"  ... and {len(results) - 10} more.")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved all predictions to {args.output}")


if __name__ == "__main__":
    main()
