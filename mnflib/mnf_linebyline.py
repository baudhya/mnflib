"""Line-by-line MNF: accumulates statistics incrementally and transforms each line as it arrives."""

import os
import numpy as np
import rasterio
from scipy.linalg import eigh
from typing import Optional, Tuple

from .data_structures import ImageStatistics, TransformDirection as TransDir, MNFResult, MNFConfig
from .utils import save_to_tiff


class Line_By_Line_MNF:
    """Memory-efficient MNF that processes and accumulates statistics one line at a time.

    Covariance matrices are built incrementally using the parallel batch update
    formula.  The eigen decomposition is recomputed after every line, so quality
    improves as more data is seen.

    For ``RUN_INVERSE`` the transform requires pre-computed statistics from a
    previous forward run; those are loaded automatically from the *stats_data*
    directory that was written during the forward pass.
    """

    _VALID_NOISE_METHODS = {"next_pixel", "three_pixel", "four_pixel", "soft_diagonal"}

    def __init__(self, image: np.ndarray, mnf_config: MNFConfig) -> None:
        self.image = np.ascontiguousarray(image.copy(), dtype=np.float32)
        self.mnf_config = mnf_config
        self.direction = mnf_config.direction
        self.bands = int(mnf_config.bands)
        self.samples = int(mnf_config.samples)
        self.lines = int(mnf_config.lines)
        self.numBandsInInv = int(mnf_config.percentageOfBandsInInverse * self.bands)

        if not (0 < self.numBandsInInv <= self.bands):
            raise ValueError(
                f"numBandsInInv must be in [1, {self.bands}], "
                f"got {self.numBandsInInv} "
                f"(percentageOfBandsInInverse={mnf_config.percentageOfBandsInInverse})"
            )
        if mnf_config.noiseMatrixCalculation not in self._VALID_NOISE_METHODS:
            raise ValueError(
                f"Unknown noiseMatrixCalculation '{mnf_config.noiseMatrixCalculation}'. "
                f"Valid options: {sorted(self._VALID_NOISE_METHODS)}"
            )

        self.basefilename = os.path.splitext(mnf_config.image_path)[0]
        self._one_samples = np.ones(self.samples, dtype=np.float32)

        # R zeroes out all but the top numBandsInInv MNF components.
        self._R = np.zeros((self.bands, self.bands), dtype=np.float32)
        for i in range(self.numBandsInInv):
            self._R[i, i] = 1.0

        self.img_stats: Optional[ImageStatistics] = None
        self.noise_stats: Optional[ImageStatistics] = None
        self._cache_prev_line: Optional[np.ndarray] = None

        self._save_folders: dict = {}
        self._create_output_folders()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> MNFResult:
        """Execute the configured transform direction(s)."""
        print("[Line_By_Line_MNF] Starting...")

        if self.direction == TransDir.RUN_INVERSE:
            self._load_stats()
            self._run_inverse_pass()
        else:
            # _process_line handles reconstruction inline for RUN_BOTH,
            # so a second _run_inverse_pass call is not needed here.
            self._run_forward_pass()

        # Persist for potential later RUN_INVERSE runs.
        if self.direction in (TransDir.RUN_FORWARD, TransDir.RUN_BOTH):
            self._save_stats()

        if self.img_stats is not None and self.img_stats.n > 0:
            _, _, eigvals = self._compute_transform_matrices() or (None, None, None)
        else:
            eigvals = None

        tag = f"{self.mnf_config.noiseMatrixCalculation}_{self.numBandsInInv}"
        out_path = os.path.join(
            self._save_folders["output_images"],
            f"_line_by_line_modified_{tag}.tif",
        )
        save_to_tiff(self.image, out_path, profile=self.mnf_config.profile)

        return MNFResult(
            eigenvalues=self.img_stats and self._last_eigvals,
            eigenvectors=self.img_stats and self._last_eigvecs,
            image_mean=self.img_stats.get_means() if self.img_stats and self.img_stats.n else None,
            noise_mean=self.noise_stats.get_means() if self.noise_stats and self.noise_stats.n else None,
            image_covariance=self.img_stats.get_cov() if self.img_stats and self.img_stats.n else None,
            noise_covariance=self.noise_stats.get_cov() if self.noise_stats and self.noise_stats.n else None,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _run_forward_pass(self) -> None:
        self.img_stats = ImageStatistics(self.bands)
        self.noise_stats = ImageStatistics(self.bands)
        self._cache_prev_line = None
        self._last_eigvals = None
        self._last_eigvecs = None

        for line_idx in range(self.lines):
            self.image[line_idx] = self._process_line(self.image[line_idx])
            if (line_idx + 1) % 100 == 0 or (line_idx + 1) == self.lines:
                print(f"[Line_By_Line_MNF] processed line {line_idx + 1}/{self.lines}")

        if self.img_stats.n > 0:
            np.savetxt(
                os.path.join(self._save_folders["eigen_data"], "_line_by_line_eigvals.dat"),
                self._last_eigvals if self._last_eigvals is not None else np.array([]),
            )

    def _process_line(self, line: np.ndarray) -> np.ndarray:
        """Update statistics with *line* then apply forward (and optionally inverse) transform."""
        self.img_stats.update_with_line(line.astype(np.float64), self.samples)
        noise_est, noise_samples = self._estimate_noise(line)
        if noise_samples > 0:
            self.noise_stats.update_with_line(noise_est.astype(np.float64), noise_samples)

        F, F_inv, eigvals = self._compute_transform_matrices() or (None, None, None)
        if F is None:
            return line.copy()

        img_mean = self.img_stats.get_means().astype(np.float32)
        mean_col = np.outer(img_mean, self._one_samples)

        forward = F.T.astype(np.float32) @ (line - mean_col)

        if self.direction == TransDir.RUN_FORWARD:
            return forward

        # RUN_BOTH: reconstruct
        filtered = self._R @ forward
        return F_inv.T.astype(np.float32) @ filtered + mean_col

    # ------------------------------------------------------------------
    # Inverse pass (standalone or second pass of RUN_BOTH)
    # ------------------------------------------------------------------

    def _run_inverse_pass(self) -> None:
        """Apply inverse transform to every line using the current (loaded) statistics."""
        result = self._compute_transform_matrices()
        if result is None:
            raise RuntimeError(
                "Cannot run inverse transform: statistics are not available. "
                "Run or load a forward pass first."
            )
        F, F_inv, _ = result
        img_mean = self.img_stats.get_means().astype(np.float32)
        mean_col = np.outer(img_mean, self._one_samples)
        invR = (F_inv.T.astype(np.float32)) @ self._R

        for line_idx in range(self.lines):
            self.image[line_idx] = invR @ self.image[line_idx] + mean_col

    # ------------------------------------------------------------------
    # Noise estimation
    # ------------------------------------------------------------------

    def _estimate_noise(self, line: np.ndarray) -> Tuple[np.ndarray, int]:
        if line.shape[1] <= 1:
            return np.zeros((self.bands, 0), dtype=np.float32), 0

        method = self.mnf_config.noiseMatrixCalculation
        prev = self._cache_prev_line

        if method == "next_pixel":
            noise_est = line[:, :-1] - line[:, 1:]
        elif method == "three_pixel":
            horiz = line[:, :-1] - line[:, 1:]
            if prev is not None:
                vert = line[:, :-1] - prev[:, :-1]
                diag = line[:, :-1] - prev[:, 1:]
                noise_est = (horiz + vert + diag) / 3.0
            else:
                noise_est = horiz
        elif method == "four_pixel":
            horiz = line[:, :-1] - line[:, 1:]
            if prev is not None:
                vert = line[:, :-1] - prev[:, :-1]
                below_horiz = prev[:, :-1] - prev[:, 1:]
                diag = line[:, :-1] - prev[:, 1:]
                noise_est = (horiz + vert + below_horiz + diag) / 4.0
            else:
                noise_est = horiz
        else:  # soft_diagonal
            if prev is not None:
                d = line[:, :-1] - prev[:, 1:]
                tau = float(np.median(np.abs(d))) * 0.25
                noise_est = np.sign(d) * np.maximum(np.abs(d) - tau, 0)
            else:
                noise_est = line[:, :-1] - line[:, 1:]

        self._cache_prev_line = line.copy()
        return noise_est, line.shape[1] - 1

    # ------------------------------------------------------------------
    # Eigen decomposition
    # ------------------------------------------------------------------

    def _compute_transform_matrices(
        self,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Compute (forward, inverse, eigenvalues) or return None if data is insufficient."""
        if self.img_stats is None or self.noise_stats is None:
            return None
        if self.img_stats.n == 0 or self.noise_stats.n == 0:
            return None

        img_cov = np.asarray(self.img_stats.get_cov(), dtype=np.float64)
        noise_cov = np.asarray(self.noise_stats.get_cov(), dtype=np.float64)
        img_cov = 0.5 * (img_cov + img_cov.T)
        noise_cov = 0.5 * (noise_cov + noise_cov.T)

        if not np.any(img_cov) or not np.any(noise_cov):
            return None

        eig_vals, eig_vecs = self._solve_eigh(noise_cov, img_cov)
        if eig_vals is None:
            return None

        idx = np.argsort(eig_vals)
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]

        try:
            inv_vecs = np.linalg.inv(eig_vecs)
        except np.linalg.LinAlgError:
            inv_vecs = np.linalg.pinv(eig_vecs)

        self._last_eigvals = eig_vals
        self._last_eigvecs = eig_vecs
        return eig_vecs, inv_vecs, eig_vals

    @staticmethod
    def _solve_eigh(
        noise_cov: np.ndarray, img_cov: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Attempt eigh with adaptive Tikhonov regularisation on img_cov."""
        # eigh(A, B) solves A·v = λ·B·v; noise in A → eigenvalues are noise/signal ratios
        try:
            return eigh(noise_cov, img_cov)
        except np.linalg.LinAlgError:
            pass

        trace = np.trace(img_cov) or 1.0
        eps = 1e-8 * trace / max(1, img_cov.shape[0])
        for _ in range(10):
            try:
                vals, vecs = eigh(noise_cov, img_cov + np.eye(img_cov.shape[0]) * eps)
                return vals, vecs
            except np.linalg.LinAlgError:
                eps *= 10.0

        # Last-resort Cholesky whitening
        try:
            L = np.linalg.cholesky(img_cov + np.eye(img_cov.shape[0]) * eps)
            Linv = np.linalg.inv(L)
            A = Linv @ noise_cov @ Linv.T
            u_vals, u_vecs = np.linalg.eigh(A)
            return u_vals, np.linalg.solve(L.T, u_vecs)
        except Exception as exc:
            print(f"[Line_By_Line_MNF] Eigen decomposition failed: {exc}")
            return None, None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_stats(self) -> None:
        stats_dir = self._save_folders["stats_data"]
        if self.img_stats and self.img_stats.n > 0:
            self.img_stats.write_to_file(os.path.join(stats_dir, "_line_by_line_image"))
        if self.noise_stats and self.noise_stats.n > 0:
            self.noise_stats.write_to_file(os.path.join(stats_dir, "_line_by_line_noise"))

    def _load_stats(self) -> None:
        stats_dir = self._save_folders["stats_data"]
        img_path = os.path.join(stats_dir, "_line_by_line_image")
        noise_path = os.path.join(stats_dir, "_line_by_line_noise")

        if not (
            os.path.exists(img_path + "_cov.pkl")
            and os.path.exists(noise_path + "_cov.pkl")
        ):
            raise FileNotFoundError(
                f"Saved statistics not found in {stats_dir}. "
                "Run a forward pass first before running inverse-only."
            )

        self.img_stats = ImageStatistics(self.bands)
        self.noise_stats = ImageStatistics(self.bands)
        self.img_stats.read_from_file(img_path)
        self.noise_stats.read_from_file(noise_path)
        self._last_eigvals = None
        self._last_eigvecs = None
        print(f"[Line_By_Line_MNF] Loaded statistics from {stats_dir}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_output_folders(self) -> None:
        image_dir = os.path.dirname(self.basefilename)
        for folder in ["output_images", "eigen_data", "stats_data"]:
            path = os.path.join(image_dir, folder)
            os.makedirs(path, exist_ok=True)
            self._save_folders[folder] = path
