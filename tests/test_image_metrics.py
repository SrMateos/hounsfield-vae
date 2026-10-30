import numpy as np
import pytest

from src.metrics.image_metrics import ImageMetrics


@pytest.fixture
def volumes():
    rng = np.random.default_rng(0)
    gt = rng.uniform(-1000, 2000, size=(32, 32, 32))
    mask = np.zeros_like(gt)
    mask[4:28, 4:28, 4:28] = 1
    return gt, mask


def test_identical_volumes(volumes):
    gt, mask = volumes
    scores = ImageMetrics().score_patient(gt, gt.copy(), mask)

    assert scores["mae"] == 0.0
    assert scores["psnr"] == np.inf
    assert scores["ms_ssim"] == pytest.approx(1.0)


def test_mae_is_in_hu_and_only_inside_mask(volumes):
    gt, mask = volumes
    pred = gt + 100.0
    # Errors outside the mask must be ignored
    pred[mask == 0] += 5000.0

    scores = ImageMetrics().score_patient(gt, pred, mask)

    assert scores["mae"] == pytest.approx(100.0)
    assert scores["ms_ssim"] < 1.0


def test_worse_prediction_scores_worse(volumes):
    gt, mask = volumes
    rng = np.random.default_rng(1)
    metrics = ImageMetrics()
    small = metrics.score_patient(gt, gt + rng.normal(0, 10, gt.shape), mask)
    large = metrics.score_patient(gt, gt + rng.normal(0, 200, gt.shape), mask)

    assert small["mae"] < large["mae"]
    assert small["psnr"] > large["psnr"]
    assert small["ms_ssim"] > large["ms_ssim"]
