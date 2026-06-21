"""Noise estimation functions and image I/O utilities."""

import numpy as np
import rasterio
from typing import Optional


def noise_added_image(image: np.ndarray, mean: float = 0.0, std_dev: float = 0.01) -> np.ndarray:
    noise = np.random.normal(mean, std_dev, image.shape).astype(image.dtype)
    return image + noise


def calculate_noise_next_pixel(image3D: np.ndarray) -> np.ndarray:
    """Horizontal diff: pixel[l,b,s] - pixel[l,b,s+1]."""
    return image3D[:, :, :-1] - image3D[:, :, 1:]


def calculate_noise_soft_threshold_diagonal_diff(image3D: np.ndarray) -> np.ndarray:
    """Soft-thresholded diagonal diff — removes signal-contaminated small differences.

    tau = MAD(diffs) / 4 adapts automatically to the noise level in the image.
    """
    d = image3D[:-1, :, :-1] - image3D[1:, :, 1:]
    tau = float(np.median(np.abs(d))) * 0.25
    return np.sign(d) * np.maximum(np.abs(d) - tau, 0)


def calculate_noise_three_pixel(image3D: np.ndarray) -> np.ndarray:
    """Average of horizontal, vertical, and diagonal diffs."""
    noise = (image3D[:-1, :, :-1] - image3D[:-1, :, 1:])   # horizontal
    noise = noise + (image3D[:-1, :, :-1] - image3D[1:, :, :-1])  # vertical
    noise = noise + (image3D[:-1, :, :-1] - image3D[1:, :, 1:])   # diagonal
    return noise / 3.0


def calculate_noise_four_pixel(image3D: np.ndarray) -> np.ndarray:
    """Average of horizontal, vertical, below-horizontal, and diagonal diffs."""
    noise = (image3D[:-1, :, :-1] - image3D[:-1, :, 1:])    # horizontal
    noise = noise + (image3D[:-1, :, :-1] - image3D[1:, :, :-1])   # vertical
    noise = noise + (image3D[1:, :, :-1] - image3D[1:, :, 1:])     # below-horizontal
    noise = noise + (image3D[:-1, :, :-1] - image3D[1:, :, 1:])    # diagonal
    return noise / 4.0


def save_to_tiff(
    image: np.ndarray,
    filename: str,
    profile: Optional[dict] = None,
) -> None:
    """Save an image array (lines, bands, samples) to a GeoTIFF.

    Preserves geospatial metadata when *profile* is supplied; falls back to a
    generic identity transform otherwise.
    """
    img_rio = image.transpose(1, 0, 2)  # → (bands, lines, samples)
    bands, height, width = img_rio.shape

    if profile is not None:
        meta = dict(profile)
        meta.update(
            driver="GTiff",
            height=height,
            width=width,
            count=bands,
            dtype=img_rio.dtype,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
    else:
        meta = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": bands,
            "dtype": img_rio.dtype,
            "transform": rasterio.transform.from_origin(0, 0, 1, 1),
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }

    with rasterio.open(filename, "w", **meta) as dst:
        dst.write(img_rio)
