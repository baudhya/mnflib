"""GeoTIFF image loader; returns arrays in (lines, bands, samples) order."""

import rasterio
import rasterio.windows
import numpy as np

from pathlib import Path
from typing import Optional

from .data_structures import ImageSubset

# Integer dtypes that must be promoted to float before processing.
_INT_TO_FLOAT = {
    np.dtype("uint16"): np.float32,
    np.dtype("uint32"): np.float32,
    np.dtype("uint64"): np.float64,
    np.dtype("int16"):  np.float32,
    np.dtype("int32"):  np.float32,
    np.dtype("int64"):  np.float64,
}


class GeotifImageLoader:
    """Load a GeoTIFF hyperspectral image into a (lines, bands, samples) array.

    Parameters
    ----------
    image_filename:
        Path to the GeoTIFF file.
    image_subset:
        Optional :class:`ImageSubset` specifying the spatial window to load.
        When *None* the full image is loaded.
    """

    def __init__(
        self,
        image_filename: str,
        image_subset: Optional[ImageSubset] = None,
        header_filename: Optional[str] = None,
    ):
        self.image_filename = Path(image_filename)
        self.image_subset = image_subset
        self.header_filename = header_filename
        self._data: Optional[np.ndarray] = None
        self.profile: Optional[dict] = None

    def get_image(self) -> np.ndarray:
        """Return the image array, loading from disk on first call."""
        if self._data is None:
            self._data = self._read_image_data()
            target_dtype = _INT_TO_FLOAT.get(self._data.dtype)
            if target_dtype is not None:
                self._data = self._data.astype(target_dtype)
        return self._data.copy()

    def _read_image_data(self) -> np.ndarray:
        if not self.image_filename.exists():
            raise FileNotFoundError(f"Image file not found: {self.image_filename}")

        with rasterio.open(self.image_filename) as src:
            if self.image_subset is not None:
                s = self.image_subset
                window = rasterio.windows.Window(
                    col_off=s.startsSamp,
                    row_off=s.startLine,
                    width=s.endSamp - s.startsSamp,
                    height=s.endLine - s.startLine,
                )
                data = src.read(window=window)
                # Update profile to reflect the subset dimensions.
                transform = src.window_transform(window)
                self.profile = dict(src.profile)
                self.profile.update(
                    height=window.height,
                    width=window.width,
                    transform=transform,
                )
            else:
                data = src.read()
                self.profile = dict(src.profile)

        # rasterio returns (bands, lines, samples); transpose to (lines, bands, samples)
        return np.transpose(data, (1, 0, 2))
