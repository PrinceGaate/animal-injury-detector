"""Script to audit prefix extraction across all image files in dataset."""

from collections import defaultdict
from pathlib import Path
from animal_injury_detector.dataset import scan_split_directory, extract_base_prefix

def main():
    data_dir = Path("Major/injuries")
    if not data_dir.exists():
        data_dir = Path("dataset/injuries")

    train_samples = scan_split_directory(data_dir / "train")
    val_samples = scan_split_directory(data_dir / "valid")
    test_samples = scan_split_directory(data_dir / "test")
    all_samples = train_samples + val_samples + test_samples

    prefix_to_files = defaultdict(list)
    prefix_to_labels = defaultdict(set)
    for p, lbl in all_samples:
        pref = extract_base_prefix(p.name)
        prefix_to_files[pref].append(p.name)
        prefix_to_labels[pref].add(lbl)

    print(f"Total samples: {len(all_samples)}")
    print(f"Total distinct prefixes: {len(prefix_to_files)}")

    multi_file = {k: v for k, v in prefix_to_files.items() if len(v) > 1}
    print(f"Prefixes with >1 files: {len(multi_file)}")

    # Check if any prefix maps to multiple labels (which would indicate mixed classes for same prefix!)
    mixed_label_prefixes = {k: prefix_to_labels[k] for k, v in prefix_to_labels.items() if len(v) > 1}
    print(f"Prefixes with mixed labels (label conflict!): {len(mixed_label_prefixes)}")
    if mixed_label_prefixes:
        for k, v in mixed_label_prefixes.items():
            print(f"  Conflict: '{k}' has labels {v}, files: {prefix_to_files[k]}")

    print("\nSample multi-file prefixes:")
    for i, (k, v) in enumerate(list(multi_file.items())[:8]):
        print(f"  Prefix '{k}' (count={len(v)}):")
        for fname in v[:3]:
            print(f"    - {fname}")

    # Inspect short or problematic prefixes
    short_prefixes = {k: v for k, v in prefix_to_files.items() if len(k) <= 4}
    print(f"\nShort prefixes (len <= 4): {len(short_prefixes)}")
    for k, v in list(short_prefixes.items())[:12]:
        print(f"  '{k}' (count={len(v)}): {v[:3]}")

if __name__ == "__main__":
    main()
