"""Integration tests: forward→inverse round-trip on synthetic data."""
import numpy as np
import pytest
from mnflib import MNF, Line_By_Line_MNF, MNFConfig, TransformDirection


def _make_config(tmp_path, direction, pct=0.5, method="next_pixel", line_by_line=False):
    img_path = str(tmp_path / "synthetic.tif")
    # Create a dummy file so output-folder creation doesn't fail.
    open(img_path, "w").close()
    return MNFConfig(
        direction=direction,
        image_path=img_path,
        bands=8,
        samples=30,
        lines=20,
        percentageOfBandsInInverse=pct,
        noiseMatrixCalculation=method,
    )


def _synthetic_image(lines=20, bands=8, samples=30, seed=0):
    rng = np.random.default_rng(seed)
    # Smooth signal + small noise → MNF should be able to reconstruct well.
    signal = rng.random((lines, bands, samples)).astype(np.float32)
    return signal


class TestMNFRoundTrip:
    @pytest.mark.parametrize("method", ["next_pixel", "three_pixel", "four_pixel"])
    def test_run_both_reduces_noise(self, tmp_path, method):
        img = _synthetic_image()
        cfg = _make_config(tmp_path, TransformDirection.RUN_BOTH, pct=0.5, method=method)
        mnf = MNF(img, cfg)
        result = mnf.run()
        assert result.eigenvalues is not None
        assert result.eigenvalues.shape == (cfg.bands,)
        # Processed image must have same shape
        assert mnf.image.shape == img.shape

    def test_run_both_full_bands_recovers_input(self, tmp_path):
        """Keeping 100 % of bands → reconstruction should be close to original."""
        img = _synthetic_image()
        cfg = _make_config(tmp_path, TransformDirection.RUN_BOTH, pct=1.0)
        mnf = MNF(img, cfg)
        mnf.run()
        np.testing.assert_allclose(mnf.image, img, atol=1e-3)

    def test_run_forward_changes_image(self, tmp_path):
        img = _synthetic_image()
        cfg = _make_config(tmp_path, TransformDirection.RUN_FORWARD)
        mnf = MNF(img, cfg)
        mnf.run()
        assert not np.allclose(mnf.image, img)

    def test_invalid_noise_method_raises(self, tmp_path):
        cfg = _make_config(tmp_path, TransformDirection.RUN_BOTH)
        cfg = MNFConfig(
            direction=TransformDirection.RUN_BOTH,
            image_path=cfg.image_path,
            bands=cfg.bands,
            samples=cfg.samples,
            lines=cfg.lines,
            percentageOfBandsInInverse=0.5,
            noiseMatrixCalculation="bogus_method",
        )
        with pytest.raises(ValueError, match="Unknown noiseMatrixCalculation"):
            MNF(_synthetic_image(), cfg)

    def test_invalid_percentage_raises(self, tmp_path):
        img_path = str(tmp_path / "x.tif")
        open(img_path, "w").close()
        with pytest.raises(ValueError, match="percentageOfBandsInInverse"):
            MNF(
                _synthetic_image(),
                MNFConfig(
                    direction=TransformDirection.RUN_BOTH,
                    image_path=img_path,
                    bands=8, samples=30, lines=20,
                    percentageOfBandsInInverse=0.0,
                    noiseMatrixCalculation="next_pixel",
                ),
            )

    def test_eigenvalues_sorted_ascending(self, tmp_path):
        img = _synthetic_image()
        cfg = _make_config(tmp_path, TransformDirection.RUN_BOTH)
        mnf = MNF(img, cfg)
        result = mnf.run()
        assert np.all(np.diff(result.eigenvalues) >= 0)

    def test_eigen_cache_not_recomputed(self, tmp_path):
        """Calling run() should compute eigen decomposition exactly once."""
        img = _synthetic_image()
        cfg = _make_config(tmp_path, TransformDirection.RUN_BOTH)
        mnf = MNF(img, cfg)
        mnf.run()
        cache_after_run = mnf._eigen_cache
        # A second call to _get_eigen() should return the same object.
        assert mnf._get_eigen() is cache_after_run


class TestLineByLineMNFRoundTrip:
    def test_run_both_produces_output(self, tmp_path):
        img = _synthetic_image()
        cfg = _make_config(tmp_path, TransformDirection.RUN_BOTH)
        lbl = Line_By_Line_MNF(img, cfg)
        result = lbl.run()
        assert lbl.image.shape == img.shape

    def test_invalid_percentage_raises(self, tmp_path):
        img_path = str(tmp_path / "x.tif")
        open(img_path, "w").close()
        with pytest.raises(ValueError):
            Line_By_Line_MNF(
                _synthetic_image(),
                MNFConfig(
                    direction=TransformDirection.RUN_BOTH,
                    image_path=img_path,
                    bands=8, samples=30, lines=20,
                    percentageOfBandsInInverse=0.0,
                    noiseMatrixCalculation="next_pixel",
                ),
            )

    def test_standalone_inverse_raises_without_stats(self, tmp_path):
        img = _synthetic_image()
        cfg = _make_config(tmp_path, TransformDirection.RUN_INVERSE)
        lbl = Line_By_Line_MNF(img, cfg)
        with pytest.raises(FileNotFoundError):
            lbl.run()
