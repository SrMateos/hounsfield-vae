import torch

from src.losses import FFL3D, KLLoss


def test_kl_is_zero_for_standard_normal():
    mu = torch.zeros(2, 3, 4, 4, 4)
    sigma = torch.ones(2, 3, 4, 4, 4)

    assert KLLoss()(mu, sigma).item() < 1e-6


def test_kl_grows_away_from_prior():
    sigma = torch.ones(2, 3, 4, 4, 4)

    assert KLLoss()(torch.full_like(sigma, 2.0), sigma) > KLLoss()(torch.full_like(sigma, 0.5), sigma)


def test_ffl3d_accepts_volumes():
    pred = torch.rand(2, 1, 16, 16, 8)
    target = torch.rand(2, 1, 16, 16, 8)

    assert FFL3D()(pred, pred).item() == 0.0
    assert FFL3D()(pred, target).item() > 0.0
