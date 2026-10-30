"""
Data handling module for SSL medical imaging project.

This module provides utilities for:
- Data transformations and augmentations
- Dataset loading and management
- Data validation and quality checks
"""

from .transforms import (
    get_vae_train_transforms,
    get_vae_val_transforms,
    get_vae_post_transforms,
    make_transform_hash,
)

from .datasets import (
    get_data_paths,
    split_data,
)

__all__ = [
    # Transforms
    "get_vae_train_transforms",
    "get_vae_val_transforms",
    "get_vae_post_transforms",
    "make_transform_hash",

    # Datasets
    "get_data_paths",
    "split_data",
]
