import time
import numpy as np
import rasterio
from functools import wraps
from typing import Optional


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"Function {func.__name__} took {time.perf_counter() - start:.4f}s")
        return result
    return wrapper


def noise_added_image(image: np.ndarray, mean: float = 0.0, std_dev: float = 0.01) -> np.ndarray:
    noise = np.random.normal(mean, std_dev, image.shape).astype(image.dtype)
    return image + noise


def calculate_noise_next_pixel(image3D: np.ndarray) -> np.ndarray:
    """Horizontal diff: pixel[l,b,s] - pixel[l,b,s+1]."""
    return image3D[:, :, :-1] - image3D[:, :, 1:]


def calculate_noise_three_pixel_with_i_pixel(image3D: np.ndarray) -> np.ndarray:
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
