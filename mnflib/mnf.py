import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.linalg import eigh

from .data_structures import ImageStatisticsFull, TransformDirection as TransDir, MNFConfig, MNFResult
from .utils import (
    timer, 
    calculate_noise_next_pixel, 
    calculate_noise_four_pixel, 
    calculate_noise_three_pixel_with_i_pixel, 
    save_to_tiff
)


# ============================================================
# Highly Optimized MNF (NumPy BLAS)
# ============================================================

class MNF:
    """
    Highly Optimized Minimum Noise Fraction (MNF) Transformation.

    Uses NumPy BLAS/LAPACK optimizations for efficient processing of hyperspectral data.
    """

    _OUTPUT_FOLDERS: List[str] = ['output_images', 'eigen_data', 'stats_data']
    
    _NOISE_CALCULATORS: Dict[str, Any] = {
        "next_pixel": calculate_noise_next_pixel,
        "four_pixel": calculate_noise_four_pixel,
        "three_pixel": calculate_noise_three_pixel_with_i_pixel
    }

    def __init__(self, image: np.ndarray, mnf_config: MNFConfig) -> None:
        """
        Initialize the MNF transformer.

        Args:
            image: Input image array of shape (lines, bands, samples).
            mnf_config: Configuration object containing MNF parameters.
        """
        # Ensure contiguous input for BLAS optimization
        self.image: np.ndarray = np.ascontiguousarray(image.copy(), dtype=np.float32)

        self.cfg: MNFConfig = mnf_config
        self.bands: int = mnf_config.bands
        self.samples: int = mnf_config.samples
        self.lines: int = mnf_config.lines
        self.direction: TransDir = mnf_config.direction

        self._final_eigen_values: Optional[np.ndarray] = None
        self._final_eigen_vectors: Optional[np.ndarray] = None

        self._save_folders: Dict[str, str] = {}

        self.basefilename: str = os.path.splitext(mnf_config.basefilename)[0]
        self.numBandsInInv: int = int(self.bands * mnf_config.percentageOfBandsInInverse)

        if not (1 <= self.numBandsInInv <= self.bands):
            raise ValueError(f"Invalid # inverse bands: {self.numBandsInInv}")

        # R matrix for selecting MNF components in inverse transform
        self._R: np.ndarray = np.zeros((self.bands, self.bands))
        np.fill_diagonal(self._R[:self.numBandsInInv, :self.numBandsInInv], 1.0)

        self._one_samples: np.ndarray = np.ones(self.samples)

        self.img_stats: ImageStatisticsFull = ImageStatisticsFull(self.bands)
        self.noise_stats: ImageStatisticsFull = ImageStatisticsFull(self.bands)

        print(f"[MNF] Inverse keeps {self.numBandsInInv} of {self.bands} bands.")

        self._create_output_folder()


    def _create_output_folder(self) -> None:
        """Create output directories for images, eigenvalues, and statistics."""
        image_dir = os.path.dirname(self.basefilename)
        
        for folder in self._OUTPUT_FOLDERS:
            path = os.path.join(image_dir, folder)
            os.makedirs(path, exist_ok=True)
            print(f"Output folder created: {path}")
            self._save_folders[folder] = path

        
    # ============================================================
    # MAIN ENTRY POINT
    # ============================================================

    def run(self) -> MNFResult:
        """
        Execute the MNF transformation pipeline based on configured direction.
        
        Handles Forward, Inverse, or Both directions. Optimize to avoid redundant
        statistic loading when running both.

        Returns:
            MNFResult object containing computed statistics and eigen-decomposition data.
        """
        print("[MNF] Starting...")

        if self.direction in (TransDir.RUN_FORWARD, TransDir.RUN_BOTH):
            self._execute_forward_transform()
        elif self.direction == TransDir.RUN_INVERSE:
            # Only load stats if we didn't just compute them
            self._load_stats()
            # Must compute matrices to get eigenvectors/values for the result object
            # even if we are only running inverse (since we need them for inverse anyway)
            # Note: _run_inverse calls _compute_mnf_matrices internally, which populates 
            # self._final_eigen_values and vectors.

        if self.direction in (TransDir.RUN_INVERSE, TransDir.RUN_BOTH):
            self._execute_inverse_transform()

        return MNFResult(
            eigenvalues=self._final_eigen_values,
            eigenvectors=self._final_eigen_vectors,
            image_mean=self.img_stats.get_means(),
            noise_mean=self.noise_stats.get_means(), # Should be zero-centered usually
            image_covariance=self.img_stats.get_cov(),
            noise_covariance=self.noise_stats.get_cov()
        )


    def _execute_forward_transform(self) -> None:
        """Execute the forward MNF transform workflow."""
        # ---------------------------------------------------------
        # FORWARD TRANSFORM
        self._compute_statistics()
        self._run_forward()
        self._save_stats()

        # Saving forward transformed images
        save_path = self._save_folders['output_images']
        # save_to_tiff(
        #     image=self.image,
        #     filename=os.path.join(save_path, "forward_transformed_without_R.tif")
        # )
        # save_to_tiff(
        #     image=self._R @ self.image,
        #     filename=os.path.join(save_path, "forward_transformed_with_R.tif")
        # )

    def _execute_inverse_transform(self) -> None:
        """Execute the inverse MNF transform workflow."""
        # ---------------------------------------------------------
        # INVERSE TRANSFORM
        # ---------------------------------------------------------
        self._run_inverse()
        save_path = self._save_folders['output_images']
        save_to_tiff(
            image=self.image,
            filename=os.path.join(save_path, f"inverse_transformed_{self.cfg.noiseMatrixCalculation}_{self.numBandsInInv}.tif")
        )


    # ============================================================
    # STATISTICS (BLAS optimized)
    # ============================================================

    def _compute_statistics(self) -> None:
        """
        Compute image & noise covariances using contiguous BLAS-ready arrays.
        """

        # Flatten to (bands, pixels)
        img_flat = self.image.transpose(1, 0, 2).reshape(self.bands, -1)
        img_flat = np.ascontiguousarray(img_flat)

        self.img_stats.update(img_flat)

        # Noise estimate
        noise_matrix_calc = self._NOISE_CALCULATORS.get(
            self.cfg.noiseMatrixCalculation, 
            calculate_noise_next_pixel
        )

        print("[MNF] Noise Calculation Method : ", noise_matrix_calc.__name__)

        noise = noise_matrix_calc(self.image)
        noise_flat = noise.transpose(1, 0, 2).reshape(self.bands, -1)
        noise_flat = np.ascontiguousarray(noise_flat)

        self.noise_stats.update(noise_flat)


    # ============================================================
    # FORWARD MNF (BLAS-optimized)
    # ============================================================

    def _run_forward(self) -> None:
        """
        Perform the actual forward MNF matrix operations.
        
        Updates self.image in-place.
        """
        means = np.ascontiguousarray(self.img_stats.get_means())
        F, _, eigvals = self._compute_mnf_matrices()

        # Save eigenvalues
        np.savetxt(os.path.join(self._save_folders['eigen_data'], "eigvals.dat"), eigvals)

        # Precompute mean expanded once (bands × samples)
        mean_matrix = means[:, None] @ self._one_samples[None, :]

        F_T = F.T  # For BLAS GEMM

        # Process each line using gemm (very fast)
        for i in range(self.lines):
            self.image[i] = F_T @ (self.image[i] - mean_matrix)


    # ============================================================
    # INVERSE MNF (BLAS-optimized)
    # ============================================================

    def _run_inverse(self) -> None:
        """
        Perform the actual inverse MNF matrix operations.
        
        Updates self.image in-place.
        """
        means = np.ascontiguousarray(self.img_stats.get_means())

        F, F_inv, _ = self._compute_mnf_matrices()

        # Precomputed inverse * R
        invR = F_inv.T @ self._R

        # Expand mean only once
        mean_matrix = means[:, None] @ self._one_samples[None, :]

        for i in range(self.lines):
            self.image[i] = invR @ self.image[i] + mean_matrix


    # ============================================================
    # MNF TRANSFORM MATRICES
    # ============================================================

    def _compute_mnf_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the generalized eigenvalue problem.
        
        Returns:
            Tuple containing (eigenvectors, inverse_eigenvectors, eigenvalues).
        """

        img_cov = np.ascontiguousarray(self.img_stats.get_cov())
        noise_cov = np.ascontiguousarray(self.noise_stats.get_cov())

        # BLAS/LAPACK generalized eigenvalue solve
        eigvals, eigvecs = eigh(noise_cov, img_cov)

        self._final_eigen_values = eigvals
        self._final_eigen_vectors = eigvecs

        # Sort low → high (high SNR first)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Pre-invert
        F_inv = np.linalg.inv(eigvecs)

        return eigvecs, F_inv, eigvals


    # ============================================================
    # SAVE STATS
    # ============================================================

    def _save_stats(self) -> None:
        """Persist computed statistics to disk."""
        self.img_stats.write_to_file(os.path.join(self._save_folders['stats_data'], "image"))
        self.noise_stats.write_to_file(os.path.join(self._save_folders['stats_data'], "noise"))

    def _load_stats(self) -> None:
        """Load statistics from disk."""
        print("[MNF] Reading covariance from file...")
        self.img_stats.read_from_file(os.path.join(self._save_folders['stats_data'], "image"))
        self.noise_stats.read_from_file(os.path.join(self._save_folders['stats_data'], "noise"))
