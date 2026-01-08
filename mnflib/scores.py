from skimage.metrics import structural_similarity as ssim
import numpy as np



def mean_squre_error(imageA, imageB):
    return np.mean((imageA - imageB) ** 2)


def root_mean_square_error(imageA, imageB):
    return np.mean((imageA - imageB) ** 2) ** 0.5


def ssim_score(imageA, imageB, full=True):
    data_range = max(imageA.max(), imageB.max()) - min(imageA.min(), imageB.min())
    score, diff = ssim(imageA, imageB, data_range=data_range, full=full)
    return score


def psnr_score(original_image, denoised_image):
    return 20 * np.log10(np.max(original_image) / mean_squre_error(original_image, denoised_image) ** 0.5)



def snr(band):
    """ mean / variance """
    return np.mean(band) / np.std(band)


class QualityMetricCalculator:
    def __init__(self, original_image, processed_image):
        """
        Initialize with original and processed images.
        Images should be in (lines, bands, samples) format.
        """
        self.original = original_image
        self.processed = processed_image
        self.lines, self.bands, self.samples = original_image.shape

    def get_mse(self):
        return mean_squre_error(self.original, self.processed)

    def get_rmse(self):
        return root_mean_square_error(self.original, self.processed)

    def get_psnr(self):
        return psnr_score(self.original, self.processed)

    def get_ssim(self):
        return ssim_score(self.original, self.processed)

    def get_variance_change(self):
        var_orig = np.var(self.original)
        var_proc = np.var(self.processed)
        return var_orig, var_proc

    def get_band_variance(self, b_idx):
        return np.var(self.processed[:, b_idx, :])

    def get_variance_stats(self):
        return {b + 1 : self.get_band_variance(b) for b in range(self.bands) }

    def get_snr_stats(self):
        """
        Return a list of dicts with SNR stats for each band.
        """
        stats = []
        for b in range(self.bands):
            snr_orig = snr(self.original[:, b, :])
            snr_proc = snr(self.processed[:, b, :])
            stats.append({
                "band": b + 1,
                "snr_original": snr_orig,
                "snr_processed": snr_proc,
                "snr_difference": snr_proc - snr_orig
            })
        return stats

    def get_band_ssim(self):
        """
        Return SSIM for each band.
        """
        res = []
        for b in range(self.bands):
            score = ssim_score(self.original[:, b, :], self.processed[:, b, :])
            res.append({"band": b + 1, "ssim": score})
        return res

    def calculate_all_metrics(self):
        """
        Calculate and return all global metrics.
        """
        var_orig, var_proc = self.get_variance_change()
        return {
            "ssim_original": self.get_ssim(),
            "mse": self.get_mse(),
            "psnr": self.get_psnr(),
            "variance_original": var_orig,
            "variance_processed": var_proc
        }

