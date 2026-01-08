import time
import numpy as np
import rasterio
from functools import wraps



def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Function {func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper


def noise_added_image(image, mean=0.0, std_dev=0.01):
    noise = np.random.normal(mean, std_dev, image.shape).astype(image.dtype)
    hsi_noisy = image + noise
    return hsi_noisy


def calculate_noise_next_pixel(image3D):
    lines, bands, samples = image3D.shape
    noise = np.zeros((lines, bands, samples-1), dtype=image3D.dtype)
    for l in range(lines):
        noise[l] = (image3D[l][:, :-1] - image3D[l][:, 1:])
    return noise


def calculate_noise_four_pixel(image3D):
    lines, bands, samples = image3D.shape
    noise = np.zeros((lines-1, bands, samples-1), dtype=image3D.dtype)
    for l in range(lines-1):
        noise[l] = image3D[l][:, :-1] - image3D[l][:, 1:] # p[][b][s] - p[l][b][s+1]
        noise[l] += (image3D[l] - image3D[l+1])[:, :-1] # p[l][b][s]
        noise[l] += image3D[l+1][:, :-1] - image3D[l+1][:, 1:] # below next
        noise[l] += image3D[l+1][:, :-1] - image3D[l+1][:, 1:] # next
    return noise / 4


def save_to_tiff(image, filename, base_meta=None):
    """
    Save specific image array to TIFF.
    If base_meta is provided, it uses it as a template.
    """
    img_rio = image.transpose(1, 0, 2)
    bands, height, width = img_rio.shape

    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": bands,
        "dtype": img_rio.dtype,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    # If we want to preserve geotransform from original, we could pass it in base_meta
    # For now, default to generic transform if not present
    if base_meta:
        meta.update(base_meta) # carefully override
    else:
        # Default identity transform if none provided
        meta["transform"] = rasterio.transform.from_origin(0, 0, 1, 1)

    with rasterio.open(filename, "w", **meta) as dst:
        dst.write(img_rio)


def calculate_noise_three_pixel_with_i_pixel(image3D):
    lines, bands, samples = image3D.shape
    noise = np.zeros((lines-1, bands, samples-1), dtype=image3D.dtype)
    for l in range(lines-1):
        noise[l] = image3D[l][:, :-1] - image3D[l][:, 1:] # next
        noise[l] += (image3D[l] - image3D[l+1])[:, :-1] # below
        noise[l] += image3D[l][:, :-1] - image3D[l+1][:, 1:] # below next
    return noise / 3