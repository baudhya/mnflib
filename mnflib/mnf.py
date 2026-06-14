import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import eigh

from .data_structures import ImageStatisticsFull, TransformDirection as TransDir, MNFConfig, MNFResult
from .utils import (
    timer,
    calculate_noise_next_pixel,
    calculate_noise_four_pixel,
    calculate_noise_three_pixel_with_i_pixel,
    calculate_noise_soft_threshold_diagonal_diff,
    save_to_tiff,
)


class MNF:
    """Minimum Noise Fraction (MNF) transform for whole-image processing.

    Image arrays throughout this class use the convention (lines, bands, samples).
    """

    _OUTPUT_FOLDERS: List[str] = ["output_images", "eigen_data", "stats_data"]

    _NOISE_CALCULATORS = {
        "next_pixel":   calculate_noise_next_pixel,
        "four_pixel":   calculate_noise_four_pixel,
        "three_pixel":  calculate_noise_three_pixel_with_i_pixel,
        "soft_diagonal": calculate_noise_soft_threshold_diagonal_diff,
    }

    def __init__(self, image: np.ndarray, mnf_config: MNFConfig) -> None:
        if not (0.0 < mnf_config.percentageOfBandsInInverse <= 1.0):
            raise ValueError(
                f"percentageOfBandsInInverse must be in (0, 1], "
                f"got {mnf_config.percentageOfBandsInInverse}"
            )
        if mnf_config.noiseMatrixCalculation not in self._NOISE_CALCULATORS:
            raise ValueError(
                f"Unknown noiseMatrixCalculation '{mnf_config.noiseMatrixCalculation}'. "
                f"Valid options: {list(self._NOISE_CALCULATORS)}"
            )

        self.image: np.ndarray = np.ascontiguousarray(image.copy(), dtype=np.float32)
        self.cfg: MNFConfig = mnf_config
        self.bands: int = mnf_config.bands
        self.samples: int = mnf_config.samples
        self.lines: int = mnf_config.lines
        self.direction: TransDir = mnf_config.direction

        self.basefilename: str = os.path.splitext(mnf_config.basefilename)[0]
        self.numBandsInInv: int = max(1, int(self.bands * mnf_config.percentageOfBandsInInverse))

        # R selects the top numBandsInInv MNF components during reconstruction.
        self._R: np.ndarray = np.zeros((self.bands, self.bands), dtype=np.float64)
        np.fill_diagonal(self._R[:self.numBandsInInv, :self.numBandsInInv], 1.0)

        self._one_samples: np.ndarray = np.ones(self.samples, dtype=np.float64)

        self.img_stats: ImageStatisticsFull = ImageStatisticsFull(self.bands)
        self.noise_stats: ImageStatisticsFull = ImageStatisticsFull(self.bands)

        # Cached eigen decomposition — computed at most once per run.
        self._eigen_cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

        self._save_folders: Dict[str, str] = {}
        self._create_output_folders()

        print(f"[MNF] Inverse keeps {self.numBandsInInv} of {self.bands} bands.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> MNFResult:
        """Execute the MNF pipeline and return eigen-decomposition results."""
        print("[MNF] Starting...")

        if self.direction in (TransDir.RUN_FORWARD, TransDir.RUN_BOTH):
            self._execute_forward_transform()
        elif self.direction == TransDir.RUN_INVERSE:
            self._load_stats()

        if self.direction in (TransDir.RUN_INVERSE, TransDir.RUN_BOTH):
            self._execute_inverse_transform()

        eigvals, _, _ = self._get_eigen()
        return MNFResult(
            eigenvalues=self._eigen_cache[2] if self._eigen_cache else None,
            eigenvectors=self._eigen_cache[0] if self._eigen_cache else None,
            image_mean=self.img_stats.get_means(),
            noise_mean=self.noise_stats.get_means(),
            image_covariance=self.img_stats.get_cov(),
            noise_covariance=self.noise_stats.get_cov(),
        )

    # ------------------------------------------------------------------
    # Forward / inverse pipelines
    # ------------------------------------------------------------------

    def _execute_forward_transform(self) -> None:
        self._compute_statistics()
        self._run_forward()
        self._save_stats()

    def _execute_inverse_transform(self) -> None:
        self._run_inverse()
        save_path = self._save_folders["output_images"]
        out_name = os.path.join(
            save_path,
            f"inverse_transformed_{self.cfg.noiseMatrixCalculation}_{self.numBandsInInv}.tif",
        )
        save_to_tiff(image=self.image, filename=out_name, profile=self.cfg.profile)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _compute_statistics(self) -> None:
        img_flat = np.ascontiguousarray(
            self.image.transpose(1, 0, 2).reshape(self.bands, -1), dtype=np.float64
        )
        self.img_stats.update(img_flat)

        noise_fn = self._NOISE_CALCULATORS[self.cfg.noiseMatrixCalculation]
        print(f"[MNF] Noise method: {noise_fn.__name__}")
        noise = noise_fn(self.image)
        noise_flat = np.ascontiguousarray(
            noise.transpose(1, 0, 2).reshape(self.bands, -1), dtype=np.float64
        )
        self.noise_stats.update(noise_flat)

    # ------------------------------------------------------------------
    # Forward transform
    # ------------------------------------------------------------------

    def _run_forward(self) -> None:
        F, _, eigvals = self._get_eigen()
        np.savetxt(
            os.path.join(self._save_folders["eigen_data"], "eigvals.dat"), eigvals
        )
        means = self.img_stats.get_means().astype(np.float64)
        mean_matrix = np.outer(means, self._one_samples)  # (bands, samples)
        F_T = F.T.astype(np.float32)
        mean_f32 = mean_matrix.astype(np.float32)
        for i in range(self.lines):
            self.image[i] = F_T @ (self.image[i] - mean_f32)

    # ------------------------------------------------------------------
    # Inverse transform
    # ------------------------------------------------------------------

    def _run_inverse(self) -> None:
        _, F_inv, _ = self._get_eigen()
        means = self.img_stats.get_means().astype(np.float64)
        mean_matrix = np.outer(means, self._one_samples)
        invR = (F_inv.T @ self._R).astype(np.float32)
        mean_f32 = mean_matrix.astype(np.float32)
        for i in range(self.lines):
            self.image[i] = invR @ self.image[i] + mean_f32

    # ------------------------------------------------------------------
    # Eigen decomposition — computed once and cached
    # ------------------------------------------------------------------

    def _get_eigen(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (eigenvectors, inverse_eigenvectors, eigenvalues), computing once."""
        if self._eigen_cache is not None:
            return self._eigen_cache

        img_cov = np.ascontiguousarray(self.img_stats.get_cov(), dtype=np.float64)
        noise_cov = np.ascontiguousarray(self.noise_stats.get_cov(), dtype=np.float64)

        # Solve: noise_cov @ v = λ · img_cov @ v
        eigvals, eigvecs = eigh(noise_cov, img_cov)

        # Sort ascending: lowest eigenvalue = lowest noise fraction = highest SNR
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        F_inv = np.linalg.inv(eigvecs)
        self._eigen_cache = (eigvecs, F_inv, eigvals)
        return self._eigen_cache

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_stats(self) -> None:
        self.img_stats.write_to_file(
            os.path.join(self._save_folders["stats_data"], "image")
        )
        self.noise_stats.write_to_file(
            os.path.join(self._save_folders["stats_data"], "noise")
        )

    def _load_stats(self) -> None:
        print("[MNF] Loading covariance from disk...")
        self.img_stats.read_from_file(
            os.path.join(self._save_folders["stats_data"], "image")
        )
        self.noise_stats.read_from_file(
            os.path.join(self._save_folders["stats_data"], "noise")
        )

    def _create_output_folders(self) -> None:
        image_dir = os.path.dirname(self.basefilename)
        for folder in self._OUTPUT_FOLDERS:
            path = os.path.join(image_dir, folder)
            os.makedirs(path, exist_ok=True)
            self._save_folders[folder] = path
