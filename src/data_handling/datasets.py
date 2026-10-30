"""
Dataset utilities and data loading functions.

This module contains functions for loading and managing datasets, including
path resolution and data splitting utilities.
"""

from pathlib import Path
from typing import List, Tuple, Union


def get_data_paths(
    data_paths: Union[Path, List[Path]],
    task1: bool = True,
    debug: bool = False
) -> Tuple[List[str], List[str], List[str]]:
    """
    Get image, CT, and mask paths from the specified data directories.

    Args:
        data_paths: Path or list of paths to data directories
        task1: If True, use MR images; if False, use CBCT images
        debug: If True, keep only the first 10 samples overall

    Returns:
        Tuple containing:
            - image_paths: List of image file paths (MR or CBCT)
            - cts_paths: List of CT file paths
            - masks_paths: List of mask file paths
    """
    if not isinstance(data_paths, list):
        data_paths = [data_paths]

    print(f"Data paths: {data_paths}")

    image_paths, cts_paths, masks_paths = [], [], []

    for data_path_item in data_paths:
        if not data_path_item.exists():
            print(f'Warning: Path {data_path_item} does not exist. Skipping.')
            continue

        subdirs = sorted([d for d in data_path_item.iterdir() if d.is_dir()])
        print(f"Len subdirs {len(subdirs)}")

        for i, subdirectory in enumerate(subdirs):
            if "overview" in str(subdirectory):
                continue

            image_name = "mr.mha" if task1 else "cbct.mha"

            image_paths.append(str(Path(subdirectory / image_name)))
            cts_paths.append(str(Path(subdirectory / "ct.mha")))
            masks_paths.append(str(Path(subdirectory / "mask.mha")))


    print(f"Found {len(image_paths)} images, {len(cts_paths)} CTs, and {len(masks_paths)} masks.")

    if debug:
        image_paths = image_paths[:10]
        cts_paths = cts_paths[:10]
        masks_paths = masks_paths[:10]
        print("Debug mode: using only the first 10 samples.")

    return image_paths, cts_paths, masks_paths


def split_data(
    data: List[dict],
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Split data into training, validation, and test sets while balancing classes.

    Args:
        data: List of data dictionaries
        train_split: Fraction of data to use for training
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        random_seed: Random seed for reproducibility
    Returns:
        Tuple containing (train_data, val_data, test_data)
    """
    import random
    from collections import defaultdict

    # Validate splits
    total = train_split + val_split + test_split
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(f"train_split + val_split + test_split must sum to 1. Got {total}")

    # Set random seed for reproducibility
    random.seed(random_seed)

    # Group data by class based on path
    class_groups = defaultdict(list)
    for item in data:
        path = item['image']
        if 'AB' in path:
            class_groups['AB'].append(item)
        elif 'HN' in path:
            class_groups['HN'].append(item)
        elif 'TH' in path:
            class_groups['TH'].append(item)
        else:
            print(f"Warning: Unrecognized class in path {path}. Skipping.")

    train_data, val_data, test_data = [], [], []

    # Split each class group and combine
    for class_name, items in class_groups.items():
        random.shuffle(items)
        n = len(items)

        train_end = int(n * train_split)
        val_end = train_end + int(n * val_split)

        train_data.extend(items[:train_end])
        val_data.extend(items[train_end:val_end])
        test_data.extend(items[val_end:])
        print(f"Class {class_name}: {n} samples split into {len(items[:train_end])} train, {len(items[train_end:val_end])} val, {len(items[val_end:])} test.")

    # Shuffle final splits
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    print(f"Total: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test.")
    return train_data, val_data, test_data
