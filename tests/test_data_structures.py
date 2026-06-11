import numpy as np
import pytest
from mnflib.data_structures import ImageStatistics, ImageStatisticsFull, TransformDirection, MNFConfig


class TestImageStatistics:
    def _make_data(self, lines=5, bands=4, samples=10, seed=0):
        rng = np.random.default_rng(seed)
        return rng.random((lines, bands, samples)).astype(np.float32)

    def test_matches_numpy_cov_single_line(self):
        rng = np.random.default_rng(1)
        line = rng.random((4, 20)).astype(np.float32)
        stats = ImageStatistics(4)
        stats.update_with_line(line, samples=20)
        # Population covariance (bias=True → divide by n).
        expected = np.cov(line, bias=True)
        np.testing.assert_allclose(stats.get_cov(), expected, rtol=1e-5)

    def test_incremental_equals_batch(self):
        """Feeding lines one-by-one must match feeding all at once."""
        rng = np.random.default_rng(2)
        data = rng.random((6, 3, 15)).astype(np.float32)  # (lines, bands, samples)

        # Incremental
        inc = ImageStatistics(3)
        for line in data:
            inc.update_with_line(line, samples=15)

        # Batch (whole image flattened) — uses population covariance (bias=True)
        full_data = data.transpose(1, 0, 2).reshape(3, -1).astype(np.float64)
        batch = ImageStatisticsFull(3)
        batch.update(full_data)  # internally uses np.cov(bias=True)

        # Both should give the same population covariance (within float tolerance)
        np.testing.assert_allclose(inc.get_cov(), batch.get_cov(), rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(inc.get_means(), batch.get_means(), rtol=1e-5)

    def test_raises_before_any_data(self):
        stats = ImageStatistics(4)
        with pytest.raises(RuntimeError):
            stats.get_cov()

    def test_round_trip_persistence(self, tmp_path):
        rng = np.random.default_rng(3)
        line = rng.random((3, 10)).astype(np.float32)
        stats = ImageStatistics(3)
        stats.update_with_line(line, 10)

        prefix = str(tmp_path / "stats")
        stats.write_to_file(prefix)

        loaded = ImageStatistics(3)
        loaded.read_from_file(prefix)

        np.testing.assert_allclose(loaded.get_cov(), stats.get_cov(), rtol=1e-6)
        np.testing.assert_allclose(loaded.get_means(), stats.get_means(), rtol=1e-6)


class TestMNFConfig:
    def test_keyword_construction(self):
        cfg = MNFConfig(
            direction=TransformDirection.RUN_BOTH,
            basefilename="/tmp/img.tif",
            bands=10,
            samples=100,
            lines=50,
            percentageOfBandsInInverse=0.2,
            noiseMatrixCalculation="next_pixel",
        )
        assert cfg.profile is None

    def test_profile_optional(self):
        cfg = MNFConfig(
            direction=TransformDirection.RUN_FORWARD,
            basefilename="/tmp/img.tif",
            bands=4,
            samples=8,
            lines=4,
            percentageOfBandsInInverse=0.5,
            noiseMatrixCalculation="three_pixel",
            profile={"crs": "EPSG:4326"},
        )
        assert cfg.profile["crs"] == "EPSG:4326"
