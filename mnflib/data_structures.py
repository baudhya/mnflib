import pickle
import numpy as np

from typing import List, Optional
from enum import Enum
from dataclasses import dataclass, field


class TransformDirection(Enum):
    RUN_BOTH = 'RUN_BOTH'
    RUN_FORWARD = 'RUN_FORWARD'
    RUN_INVERSE = 'RUN_INVERSE'


@dataclass
class MNFConfig:
    direction: TransformDirection
    basefilename: str
    bands: int
    samples: int
    lines: int
    percentageOfBandsInInverse: float
    noiseMatrixCalculation: str
    # Rasterio profile dict for preserving geospatial metadata in outputs.
    profile: Optional[dict] = None


@dataclass
class MNFResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    image_mean: np.ndarray
    noise_mean: np.ndarray
    image_covariance: np.ndarray
    noise_covariance: np.ndarray


class HyspexHeader:
    def __init__(self):
        self.samples: int = 0
        self.bands: int = 0
        self.lines: int = 0
        self.offset: int = 0
        self.wlens: List[float] = []
        self.datatype: int = 0


class ImageSubset:
    def __init__(self):
        self.startsSamp: int = 0
        self.endSamp: int = 0
        self.startLine: int = 0
        self.endLine: int = 0
        self.startBand: int = 0
        self.endBand: int = 0


class ImageStatistics:
    """Incremental (online) per-line covariance accumulator.

    Uses the parallel/batch update formula so statistics can be merged one
    line at a time without storing all data in memory.

    ``get_cov()`` returns the population covariance (divides accumulated
    scatter by total pixel count *n*).
    """

    def __init__(self, bands: int):
        self.bands: int = bands
        self.n: int = 0
        self.C: np.ndarray = np.zeros((bands, bands), dtype=np.float64)
        self.means: np.ndarray = np.zeros(bands, dtype=np.float64)

    def get_means(self) -> np.ndarray:
        return self.means

    def get_cov(self) -> np.ndarray:
        if self.n == 0:
            raise RuntimeError("No data accumulated yet.")
        return self.C / self.n

    def update_with_line(self, line: np.ndarray, samples: int) -> None:
        """Merge a new batch of *samples* pixels into the running statistics."""
        if samples <= 0:
            return
        old_n = self.n
        self.n += samples

        line = line.astype(np.float64, copy=False)
        line_mean = line.mean(axis=1)                    # (bands,)
        line_centered = line - line_mean[:, None]        # (bands, samples)
        self.C += line_centered @ line_centered.T        # within-batch scatter

        mean_delta = line_mean - self.means
        self.means += (samples / self.n) * mean_delta
        # between-group correction (parallel update formula)
        self.C += (samples * old_n / self.n) * np.outer(mean_delta, mean_delta)

    def write_to_file(self, location: str) -> None:
        with open(location + "_cov.pkl", "wb") as f:
            pickle.dump(self.get_cov(), f)
        with open(location + "_mean.pkl", "wb") as f:
            pickle.dump(self.means, f)
        np.savetxt(location + "_cov.txt", self.get_cov())
        np.savetxt(location + "_mean.txt", self.means)

    def read_from_file(self, location: str) -> None:
        """Load statistics previously saved by ``write_to_file``.

        The stored covariance is already normalised (divided by n).  Setting
        ``n = 1`` makes ``get_cov()`` return it unchanged.
        """
        with open(location + "_cov.pkl", "rb") as f:
            self.C = pickle.load(f)
        with open(location + "_mean.pkl", "rb") as f:
            self.means = pickle.load(f)
        self.n = 1  # C already holds the normalised covariance


class ImageStatisticsFull:
    """Whole-image covariance computed in one shot via ``np.cov``.

    ``get_cov()`` returns the stored matrix directly (no division).
    """

    def __init__(self, bands: int):
        self.C: np.ndarray = np.zeros((bands, bands), dtype=np.float64)
        self.means: np.ndarray = np.zeros(bands, dtype=np.float64)

    def get_means(self) -> np.ndarray:
        return self.means

    def get_cov(self) -> np.ndarray:
        return self.C

    def update(self, image: np.ndarray) -> None:
        """Compute statistics from a (bands, pixels) array.

        Uses the population (biased) covariance (divides by n) to match the
        accumulator used by :class:`ImageStatistics`.
        """
        self.means = np.average(image, axis=1)
        self.C = np.cov(image, bias=True)  # bias=True → divide by n

    def write_to_file(self, location: str) -> None:
        with open(location + "_cov.pkl", "wb") as f:
            pickle.dump(self.C, f)
        with open(location + "_mean.pkl", "wb") as f:
            pickle.dump(self.means, f)
        np.savetxt(location + "_cov.txt", self.C)
        np.savetxt(location + "_mean.txt", self.means)

    def read_from_file(self, location: str) -> None:
        with open(location + "_cov.pkl", "rb") as f:
            self.C = pickle.load(f)
        with open(location + "_mean.pkl", "rb") as f:
            self.means = pickle.load(f)
