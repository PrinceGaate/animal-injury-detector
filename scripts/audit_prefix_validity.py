"""Audit whether extract_base_prefix identifies identical source images or distinct subjects."""

from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict
from animal_injury_detector.dataset import scan_split_directory, extract_base_prefix

def main():
    data_dir = Path("Major/injuries")
    if not data_dir.exists():
        data_dir = Path("dataset/injuries")

    all_samples = (
        scan_split_directory(data_dir / "train")
        + scan_split_directory(data_dir / "valid")
        + scan_split_directory(data_dir / "test")
    )

    groups = defaultdict(list)
    for p, lbl in all_samples:
        pref = extract_base_prefix(p.name)
        groups[pref].append((p, lbl))

    sample_groups = ["dogs_001", "dogs_003", "HEALTHY_058", "-2", "-", "images"]

    print("=================================================================")
    print("DETAILED AUDIT OF EXTRACT_BASE_PREFIX IMAGE GROUPS")
    print("=================================================================")

    for g in sample_groups:
        files = groups[g]
        print(f"\nGroup '{g}': {len(files)} files")
        img_hashes = []
        for p, lbl in files:
            with Image.open(p) as im:
                arr = np.array(im.convert("RGB").resize((64, 64)))
                mean_rgb = arr.mean(axis=(0, 1))
                print(f"  - {p.name[:48]}... | Label: {lbl} | Size: {im.size} | Mean RGB: {np.round(mean_rgb, 1)}")

if __name__ == "__main__":
    main()
