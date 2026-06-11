import numpy as np
from skimage.metrics import structural_similarity as ssim


def mean_squared_error(image_a: np.ndarray, image_b: np.ndarray) -> float:
    return float(np.mean((image_a - image_b) ** 2))


def root_mean_squared_error(image_a: np.ndarray, image_b: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(image_a, image_b)))


def ssim_score(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Compute SSIM between two 2-D spatial images (lines × samples)."""
    data_range = float(
        max(image_a.max(), image_b.max()) - min(image_a.min(), image_b.min())
    )
    min_dim = min(image_a.shape[-2], image_a.shape[-1])
    # win_size must be odd and ≤ the smallest spatial dimension.
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    score, _ = ssim(image_a, image_b, data_range=data_range, full=True, win_size=win_size)
    return float(score)


def psnr_score(original: np.ndarray, denoised: np.ndarray) -> float:
    rmse = root_mean_squared_error(original, denoised)
    if rmse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(np.max(original) / rmse))


def snr(band: np.ndarray) -> float:
    """Signal-to-noise ratio defined as mean / std."""
    std = np.std(band)
    if std == 0.0:
        return float("inf")
    return float(np.mean(band) / std)


class QualityMetricCalculator:
    """Compute image quality metrics between an original and processed image.

    Both images must be in (lines, bands, samples) format.
    """

    def __init__(self, original_image: np.ndarray, processed_image: np.ndarray):
        if original_image.shape != processed_image.shape:
            raise ValueError(
                f"Shape mismatch: original {original_image.shape} vs "
                f"processed {processed_image.shape}"
            )
        self.original = original_image
        self.processed = processed_image
        self.lines, self.bands, self.samples = original_image.shape

    def get_mse(self) -> float:
        return mean_squared_error(self.original, self.processed)

    def get_rmse(self) -> float:
        return root_mean_squared_error(self.original, self.processed)

    def get_psnr(self) -> float:
        return psnr_score(self.original, self.processed)

    def get_ssim(self) -> float:
        """Mean per-band SSIM across all bands (lines × samples slices)."""
        scores = [
            ssim_score(self.original[:, b, :], self.processed[:, b, :])
            for b in range(self.bands)
        ]
        return float(np.mean(scores))

    def get_variance_change(self):
        return float(np.var(self.original)), float(np.var(self.processed))

    def get_band_variance(self, b_idx: int) -> float:
        return float(np.var(self.processed[:, b_idx, :]))

    def get_variance_stats(self) -> dict:
        return {b + 1: self.get_band_variance(b) for b in range(self.bands)}

    def get_snr_stats(self) -> list:
        stats = []
        for b in range(self.bands):
            orig_snr = snr(self.original[:, b, :])
            proc_snr = snr(self.processed[:, b, :])
            stats.append(
                {
                    "band": b + 1,
                    "snr_original": orig_snr,
                    "snr_processed": proc_snr,
                    "snr_difference": proc_snr - orig_snr,
                }
            )
        return stats

    def get_band_ssim(self) -> list:
        return [
            {"band": b + 1, "ssim": ssim_score(self.original[:, b, :], self.processed[:, b, :])}
            for b in range(self.bands)
        ]

    def calculate_all_metrics(self) -> dict:
        var_orig, var_proc = self.get_variance_change()
        return {
            "ssim": self.get_ssim(),
            "mse": self.get_mse(),
            "psnr": self.get_psnr(),
            "variance_original": var_orig,
            "variance_processed": var_proc,
        }
