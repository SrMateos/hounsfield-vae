"""
Data transformations for VAE training and validation.

This module contains all the data augmentation and preprocessing transformations
used for training and validation of the variational autoencoder.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from monai.transforms import (Compose, LoadImaged,
                              RandAdjustContrastd, RandAffined,
                              RandGaussianNoised,
                              RandSpatialCropd, ScaleIntensityRanged,
                              SpatialPadd, EnsureTyped,
                              Invertd, Lambdad)

from src.constants import A_MAX_HU, A_MIN_HU

def unnormalize(x):
    return x * (A_MAX_HU - A_MIN_HU) + A_MIN_HU


def make_transform_hash(patch_size):
    """
    Build a ``hash_transform`` function for ``PersistentDataset``.

    MONAI's ``json_hashing`` cannot serialize transform objects and falls back to
    hashing their class names, so changing the HU window or the patch size would
    silently reuse a stale cache. This hash covers the source of this module, the
    patch size and the HU window instead.
    """
    def _hash(_hashable_transforms) -> bytes:
        key = json.dumps({
            "source": Path(__file__).read_text(),
            "patch_size": list(patch_size),
            "hu_window": [A_MIN_HU, A_MAX_HU],
        })
        return hashlib.sha256(key.encode("utf-8")).hexdigest().encode("utf-8")

    return _hash

def get_vae_train_transforms(patch_size=(64,)*3):
    """
    Get training transforms for VAE model.

    Args:
        patch_size (tuple): Size of the patches to extract (default: (64, 64, 64))

    Returns:
        Compose: MONAI compose object with training transforms
    """
    return Compose([
        LoadImaged(keys=("image", "mask"), image_only=True, ensure_channel_first=True),
        SpatialPadd(keys=("image", "mask"), method="symmetric", spatial_size=patch_size),
        ScaleIntensityRanged(
            keys=("image"),
            a_min=A_MIN_HU, # A broad range for different organ types https://radiopaedia.org/articles/windowing-ct
            a_max=A_MAX_HU,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        RandSpatialCropd(
            keys=("image", "mask"),
            roi_size=patch_size,
            random_size=False,
        ),
        # RandFlip(keys=("image", "mask"), prob=0.1, spatial_axis=0),
        # RandFlip(keys=("image", "mask"), prob=0.1, spatial_axis=1),
        # RandFlip(keys=("image", "mask"), prob=0.1, spatial_axis=2),
        RandAffined(
            keys=("image", "mask"),
            prob=0.1,
            rotate_range=(np.pi/12, np.pi/12, np.pi/12),  # Slight rotations
            scale_range=(0.1, 0.1, 0.1),  # Slight scaling
            mode=('bilinear', "nearest"),  # Interpolation for image
            padding_mode='border'
        ),
        RandAdjustContrastd(keys="image", prob=0.1, gamma=(0.7, 1.3)),
        RandGaussianNoised(
            keys="image",
            prob=0.1,
            mean=0.0,
            std=0.05 * (1.0 - 0.0)  # Adjust std to normalized range
        ),
        ScaleIntensityRanged(
            keys="image",
            a_min=0.0, a_max=1.0,   # rango de entrada esperado
            b_min=0.0, b_max=1.0,   # identidad
            clip=True               # recorta fuera de [0,1]
        ),
        EnsureTyped(keys=("image", "mask"), dtype=np.float32),
    ])


def get_vae_val_transforms(patch_size=(64,)*3):
    """
    Get validation transforms for VAE model.

    Args:
        patch_size (tuple): Size of the patches to extract (default: (64, 64, 64))

    Returns:
        Compose: MONAI compose object with validation transforms
    """
    return Compose([
        LoadImaged(keys=("image", "mask"), image_only=False, ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=("image"),
            a_min=A_MIN_HU, # A broad range for different organ types https://radiopaedia.org/articles/windowing-ct
            a_max=A_MAX_HU,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        SpatialPadd(keys=("image", "mask"), method="symmetric", spatial_size=patch_size),
        EnsureTyped(keys=("image", "mask"), dtype=np.float32),
    ])


def get_vae_post_transforms(val_tf):
    """
    Deshace SpatialPadd y ScaleIntensityRanged sobre:
        "pred"  reconstrucción del AE
        "image" opcional: GT en HU
    """
    return Compose([
        # reconstructed image and mask
        Invertd(
            keys=["pred", "mask"],
            transform=val_tf,
            orig_keys=["image", "mask"],
            meta_keys=["image_meta_dict", "mask_meta_dict"],
            nearest_interp=[False, True],
        ),
        # ground-truth
        Invertd(
            keys=["image"],
            transform=val_tf,
            orig_keys=["image"],
            meta_keys=["image_meta_dict"],
            nearest_interp=False,
        ),
        Lambdad(
            keys=["pred", "image"],
            func=unnormalize,
        ),
        EnsureTyped(keys=["pred", "image", "mask"])
    ])
