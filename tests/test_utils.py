import numpy as np
import pytest
from mnflib.utils import (
    calculate_noise_next_pixel,
    calculate_noise_three_pixel,
    calculate_noise_four_pixel,
    noise_added_image,
)


@pytest.fixture
def cube():
    """Deterministic (lines, bands, samples) float32 cube."""
    rng = np.random.default_rng(0)
    return rng.random((10, 4, 8), dtype=np.float32)


class TestNextPixel:
    def test_shape(self, cube):
        out = calculate_noise_next_pixel(cube)
        lines, bands, samples = cube.shape
        assert out.shape == (lines, bands, samples - 1)

    def test_values(self):
        arr = np.arange(12, dtype=np.float32).reshape(1, 1, 12)
        out = calculate_noise_next_pixel(arr)
        np.testing.assert_allclose(out, np.full((1, 1, 11), -1.0))


class TestThreePixel:
    def test_shape(self, cube):
        out = calculate_noise_three_pixel(cube)
        lines, bands, samples = cube.shape
        assert out.shape == (lines - 1, bands, samples - 1)

    def test_uniform_image_gives_zero_noise(self):
        arr = np.ones((5, 3, 10), dtype=np.float32)
        out = calculate_noise_three_pixel(arr)
        np.testing.assert_allclose(out, 0.0)


class TestFourPixel:
    def test_shape(self, cube):
        out = calculate_noise_four_pixel(cube)
        lines, bands, samples = cube.shape
        assert out.shape == (lines - 1, bands, samples - 1)

    def test_uniform_image_gives_zero_noise(self):
        arr = np.ones((5, 3, 10), dtype=np.float32)
        out = calculate_noise_four_pixel(arr)
        np.testing.assert_allclose(out, 0.0)

    def test_four_distinct_terms(self):
        """Verify all four differences contribute — not the same expression twice."""
        rng = np.random.default_rng(42)
        arr = rng.random((3, 2, 6), dtype=np.float32)
        horiz = arr[:-1, :, :-1] - arr[:-1, :, 1:]
        vert  = arr[:-1, :, :-1] - arr[1:, :, :-1]
        below = arr[1:,  :, :-1] - arr[1:, :, 1:]
        diag  = arr[:-1, :, :-1] - arr[1:, :, 1:]
        expected = (horiz + vert + below + diag) / 4.0
        np.testing.assert_allclose(calculate_noise_four_pixel(arr), expected)


class TestNoiseAddedImage:
    def test_same_shape_and_dtype(self, cube):
        noisy = noise_added_image(cube, std_dev=0.01)
        assert noisy.shape == cube.shape
        assert noisy.dtype == cube.dtype

    def test_noise_is_applied(self, cube):
        noisy = noise_added_image(cube, std_dev=0.1)
        assert not np.allclose(noisy, cube)
