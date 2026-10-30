import pytest
import torch
from torch.nn import L1Loss

from src.data_handling import make_transform_hash
from src.training.trainer import Trainer


def _bare_trainer(**attrs):
    """Trainer without __init__ (which needs MLflow and data)."""
    trainer = Trainer.__new__(Trainer)
    for key, value in attrs.items():
        setattr(trainer, key, value)
    return trainer


def test_l1_is_computed_even_when_its_weight_is_zero():
    # Regression: with pixel_wise_weight = 0 the validation L1 used for model
    # selection and the LR scheduler was always 0.
    trainer = _bare_trainer(
        l1_loss=L1Loss(),
        pixel_wise_weight=0.0,
        perceptual_weight=0.0,
        ssim_weight=0.0,
        focal_frequency_weight=0.0,
    )
    outputs = torch.zeros(1, 1, 8, 8, 8)
    images = torch.ones(1, 1, 8, 8, 8)

    losses = trainer._reconstruction_losses(outputs, images)

    assert losses["pixel_wise"].item() == pytest.approx(1.0)
    assert isinstance(losses["total"], torch.Tensor)
    assert losses["total"].item() == 0.0


def test_cyclical_kl_annealing():
    trainer = _bare_trainer(
        num_epochs=100, kl_annealing_cycles=2, kl_annealing_ratio=0.5, max_kl_weight=1e-4,
    )
    weights = []
    for epoch in (0, 25, 49, 50, 75):
        trainer._update_kl_weight(epoch)
        weights.append(trainer.kl_weight)

    assert weights[0] == 0.0
    assert weights[1] == pytest.approx(1e-4)  # end of the ramp
    assert weights[2] == pytest.approx(1e-4)  # plateau
    assert weights[3] == 0.0                  # second cycle restarts
    assert weights[4] == pytest.approx(1e-4)


def test_transform_hash_depends_on_patch_size():
    assert make_transform_hash((64, 64, 64))([]) != make_transform_hash((96, 96, 96))([])
    assert make_transform_hash((64, 64, 64))([]) == make_transform_hash((64, 64, 64))([])
