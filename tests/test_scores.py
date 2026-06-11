import numpy as np
import pytest
from mnflib.scores import (
    mean_squared_error,
    root_mean_squared_error,
    psnr_score,
    snr,
    QualityMetricCalculator,
)


def _identical_images():
    rng = np.random.default_rng(0)
    img = rng.random((10, 4, 20), dtype=np.float32)
    return img, img.copy()


class TestBasicMetrics:
    def test_mse_identical_is_zero(self):
        a, b = _identical_images()
        assert mean_squared_error(a, b) == pytest.approx(0.0)

    def test_rmse_identical_is_zero(self):
        a, b = _identical_images()
        assert root_mean_squared_error(a, b) == pytest.approx(0.0)

    def test_psnr_identical_is_inf(self):
        a, b = _identical_images()
        assert psnr_score(a, b) == float("inf")

    def test_psnr_decreases_with_noise(self):
        rng = np.random.default_rng(1)
        orig = rng.random((5, 3, 10), dtype=np.float32)
        noisy_low  = orig + rng.normal(0, 0.01, orig.shape).astype(np.float32)
        noisy_high = orig + rng.normal(0, 0.10, orig.shape).astype(np.float32)
        assert psnr_score(orig, noisy_low) > psnr_score(orig, noisy_high)

    def test_snr_constant_band_is_inf(self):
        band = np.ones((5, 5), dtype=np.float32)
        assert snr(band) == float("inf")

    def test_snr_positive_for_positive_signal(self):
        rng = np.random.default_rng(2)
        band = rng.random((5, 5)).astype(np.float32) + 1.0  # all positive
        assert snr(band) > 0


class TestQualityMetricCalculator:
    def test_shape_mismatch_raises(self):
        a = np.ones((5, 4, 10), dtype=np.float32)
        b = np.ones((5, 4, 11), dtype=np.float32)
        with pytest.raises(ValueError, match="Shape mismatch"):
            QualityMetricCalculator(a, b)

    def test_calculate_all_metrics_keys(self):
        rng = np.random.default_rng(3)
        img = rng.random((4, 3, 8), dtype=np.float32)
        calc = QualityMetricCalculator(img, img.copy())
        metrics = calc.calculate_all_metrics()
        assert set(metrics) == {"ssim", "mse", "psnr", "variance_original", "variance_processed"}

    def test_variance_stats_length(self):
        rng = np.random.default_rng(4)
        img = rng.random((4, 5, 8), dtype=np.float32)
        calc = QualityMetricCalculator(img, img.copy())
        var = calc.get_variance_stats()
        assert len(var) == 5  # one per band
