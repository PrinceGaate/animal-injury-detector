"""Analyze data leakage in original dataset splits and generate leakage-resistant statistics."""

import json
from pathlib import Path

from animal_injury_detector.dataset import (
    create_leakage_resistant_grouped_splits,
    extract_base_prefix,
    scan_split_directory,
)


def analyze_leakage() -> None:
    data_dir = Path("Major/injuries")
    if not data_dir.exists():
        data_dir = Path("dataset/injuries")

    train_samples = scan_split_directory(data_dir / "train")
    val_samples = scan_split_directory(data_dir / "valid")
    test_samples = scan_split_directory(data_dir / "test")

    all_samples = train_samples + val_samples + test_samples

    tr_prefixes = {extract_base_prefix(p.name) for p, _ in train_samples}
    val_prefixes = {extract_base_prefix(p.name) for p, _ in val_samples}
    te_prefixes = {extract_base_prefix(p.name) for p, _ in test_samples}

    leakage_stats = {
        "original_splits": {
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "train_distinct_subjects": len(tr_prefixes),
            "val_distinct_subjects": len(val_prefixes),
            "test_distinct_subjects": len(te_prefixes),
            "train_val_overlap_count": len(tr_prefixes.intersection(val_prefixes)),
            "train_test_overlap_count": len(tr_prefixes.intersection(te_prefixes)),
            "val_test_overlap_count": len(val_prefixes.intersection(te_prefixes)),
        }
    }

    # Generate grouped split
    lr_train, lr_val, lr_test = create_leakage_resistant_grouped_splits(
        all_samples, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    lr_tr_p = {extract_base_prefix(p.name) for p, _ in lr_train}
    lr_val_p = {extract_base_prefix(p.name) for p, _ in lr_val}
    lr_te_p = {extract_base_prefix(p.name) for p, _ in lr_test}

    leakage_stats["leakage_resistant_grouped_splits"] = {
        "train_samples": len(lr_train),
        "val_samples": len(lr_val),
        "test_samples": len(lr_test),
        "train_healthy": sum(1 for _, l in lr_train if l == 0),
        "train_injured": sum(1 for _, l in lr_train if l == 1),
        "val_healthy": sum(1 for _, l in lr_val if l == 0),
        "val_injured": sum(1 for _, l in lr_val if l == 1),
        "test_healthy": sum(1 for _, l in lr_test if l == 0),
        "test_injured": sum(1 for _, l in lr_test if l == 1),
        "train_val_overlap_count": len(lr_tr_p.intersection(lr_val_p)),
        "train_test_overlap_count": len(lr_tr_p.intersection(lr_te_p)),
        "val_test_overlap_count": len(lr_val_p.intersection(lr_te_p)),
    }

    out_p = Path("results/dataset_leakage_audit.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(leakage_stats, f, indent=2)
    print(f"Saved dataset leakage audit to {out_p}")

    manifest_p = Path("results/group_disjoint_split_manifest.json")
    manifest = {
        "random_seed": 42,
        "split_ratios": {"train": 0.70, "valid": 0.15, "test": 0.15},
        "groups": {
            "train_groups": sorted(list(lr_tr_p)),
            "val_groups": sorted(list(lr_val_p)),
            "test_groups": sorted(list(lr_te_p)),
            "train_group_count": len(lr_tr_p),
            "val_group_count": len(lr_val_p),
            "test_group_count": len(lr_te_p),
        },
        "samples": {
            "train": [{"path": str(p.relative_to(data_dir.parent.parent)), "label": l, "prefix": extract_base_prefix(p.name)} for p, l in lr_train],
            "valid": [{"path": str(p.relative_to(data_dir.parent.parent)), "label": l, "prefix": extract_base_prefix(p.name)} for p, l in lr_val],
            "test": [{"path": str(p.relative_to(data_dir.parent.parent)), "label": l, "prefix": extract_base_prefix(p.name)} for p, l in lr_test],
        },
        "audit": {
            "train_val_overlap_count": len(lr_tr_p.intersection(lr_val_p)),
            "train_test_overlap_count": len(lr_tr_p.intersection(lr_te_p)),
            "val_test_overlap_count": len(lr_val_p.intersection(lr_te_p)),
        },
    }
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved split manifest to {manifest_p}")


if __name__ == "__main__":
    analyze_leakage()
