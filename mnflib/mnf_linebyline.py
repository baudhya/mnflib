import os
import numpy as np
import rasterio
from scipy.linalg import eigh

from .data_structures import ImageStatistics, TransformDirection as TransDir, MNFResult


def is_spd(Mat, tol=1e-12):
    Mat = np.asarray(Mat)
    if not np.allclose(Mat, Mat.T, atol=1e-10):
        return False
    try:
        vals = np.linalg.eigvalsh(Mat)
        return np.all(vals > tol)
    except Exception:
        return False




class Line_By_Line_MNF():
    def __init__(self, image, mnf_config):
        self._image = np.ascontiguousarray(image, dtype=np.float32)
        self.image = self._image.copy()
        self.mnf_config = mnf_config
        self.direction = mnf_config.direction
        self.basefilename = mnf_config.basefilename
        self.bands = int(mnf_config.bands)
        self.samples = int(mnf_config.samples)
        self.lines = int(mnf_config.lines)
        self.numBandsInInv = int(mnf_config.percentageOfBandsInInverse * self.bands)

        self._one_samples = None
        self._R = None

        self.img_stats = None
        self.noise_stats = None
        self.__save_folder = {}

        self.__final_eigen_values = None
        self.__final_eigen_vectors = None

        self.cache_prev_line = None

        self._init()
        self.__create_output_folder()


    def __create_output_folder(self):
        """Create an output folder in the same directory as the image"""
        image_dir = os.path.dirname(self.basefilename)
        
        output_folder = ['output_images', 'eigen_data', 'stats_data']
        for folder in output_folder:
            path = os.path.join(image_dir, folder)
            os.makedirs(path, exist_ok=True)
            print(f"Output folder created: {path}")
            self.__save_folder[folder] = path

    def _init(self):
        print(f"Total Number of Bands in Inverse Transformation : {self.numBandsInInv}")
        if not (0 < self.numBandsInInv <= self.bands):
            raise ValueError(f"Number of inverse band should be in between 0 and {self.bands}")

        # get filename without extension
        self.basefilename = os.path.splitext(self.basefilename)[0]
        self._one_samples = np.ones(self.samples, dtype=np.float32)
        self._R = np.zeros((self.bands, self.bands), dtype=np.float32)
        for i in range(self.numBandsInInv):
            self._R[i, i] = 1.0

    def run(self):
        print("Starting Line by Line MNF ..............")
        self.img_stats = ImageStatistics(self.bands)
        self.noise_stats = ImageStatistics(self.bands)

        for line_idx in range(self.lines):
            self.image[line_idx] = self.run_one_line(self.image[line_idx])

            if (line_idx + 1) % 100 == 0 or (line_idx + 1) == self.lines:
                print(f"[Line_By_Line_MNF] processed line {line_idx+1}/{self.lines}")

        # writing statistics into files
        self.img_stats.write_to_file(os.path.join(self.__save_folder['stats_data'], "_line_by_line_image"))
        self.noise_stats.write_to_file(os.path.join(self.__save_folder['stats_data'], "_line_by_line_noise"))

        # writing eigen values into files (if computed)
        if self.__final_eigen_values is not None:
            np.savetxt(os.path.join(self.__save_folder['eigen_data'], "_line_by_line_eigvals.dat"), self.__final_eigen_values)

        # self.save_img()
        self.save_img(name=os.path.join(self.__save_folder['output_images'], f"_line_by_line_modified_{self.mnf_config.noiseMatrixCalculation}_{self.numBandsInInv}.tif"))

        return MNFResult(
            eigenvalues=self.__final_eigen_values,
            eigenvectors=self.__final_eigen_vectors,
            image_mean=self.img_stats.get_means(),
            noise_mean=self.noise_stats.get_means(), # noise usually assumed zero mean but calculated
            image_covariance=self.img_stats.get_cov(),
            noise_covariance=self.noise_stats.get_cov()
        )

    def run_one_line(self, line):
        """
        line: ndarray (bands, samples)
        returns processed line (bands, samples)
        """
        # Update statistics with this line
        self.img_stats.update_with_line(line, self.samples)
        noise_est, noise_sample = self.estimate_noise(line)
        self.noise_stats.update_with_line(noise_est, noise_sample)

        img_mean = self.img_stats.get_means()  # shape (bands,)
        # get transform matrices (forward, inverse) and eigenvalues
        forwardTransf, inverseTransf, eigvals = self.mnf_get_transf_matrix()

        # remove mean from line: line - outer(mean, ones)
        line_zero_mean = line - np.outer(img_mean, self._one_samples)

        # If transforms are not yet available (e.g., on first lines), just return original (or mean-subtracted-add-back)
        if forwardTransf is None or inverseTransf is None:
            return line.copy()

        # forward coefficients: forwardTransf.T @ line_zero_mean
        # (this matches original use of trans_a=1 with forwardTransf)
        submatr = forwardTransf.T.dot(line_zero_mean)

        if self.direction == TransDir.RUN_FORWARD:
            # Return forward coefficients (same as original code)
            return submatr
        else:
            # Perform inverse reconstruction keeping only first numBandsInInv components
            # Apply R (zero out other components)
            submatr_filtered = self._R.dot(submatr)  # (bands, samples)

            # reconstruct: inverseTransf.T @ submatr_filtered
            recon = inverseTransf.T.dot(submatr_filtered)

            # add mean back
            recon_plus_mean = recon + np.outer(img_mean, self._one_samples)

            return recon_plus_mean

    def estimate_noise(self, line):
        if line.shape[1] <= 1:
            return np.zeros((self.bands, 0), dtype=np.float32), 0

        method = self.mnf_config.noiseMatrixCalculation
        
        if method == "next_pixel":
            noise_est = self._estimate_noise_next_pixel(line)
        else:
            # Default to Next + Diagonal (covers "three_pixel", "four_pixel", "next_diagonal" etc.)
            noise_est = self._estimate_noise_next_diagonal(line)

        noise_samples = line.shape[1] - 1

        # Update cache (common for all methods that use history)
        self.cache_prev_line = line.copy()

        return noise_est, noise_samples

    def _estimate_noise_next_pixel(self, line):
        """
        Noise = Current_Pixel - Next_Pixel (Horizontal only)
        """
        return (line[:, :-1] - line[:, 1:])

    def _estimate_noise_next_diagonal(self, line):
        """
        Noise = Average of:
          1. Current_Pixel - Next_Pixel (Horizontal)
          2. Current_Pixel - PrevLine_Next_Pixel (Diagonal)
        """
        # Horizontal diff: current[b, s] - current[b, s+1]
        horiz_diff = (line[:, :-1] - line[:, 1:])
        
        if self.cache_prev_line is not None:
            # Diagonal diff: current[b, s] - prev[b, s+1]
            diag_diff = (line[:, :-1] - self.cache_prev_line[:, 1:])
            virt_diff = (line[:, :-1] - self.cache_prev_line[:, :-1])

            
            # Average of 2 directions
            return (horiz_diff + diag_diff + virt_diff) / 3.0
        else:
            # Fallback for first line
            return horiz_diff

    def mnf_get_transf_matrix(self):
        """
        Compute forward transform and its inverse. Handles failures and regularizes B (img_cov).
        Returns (forwardTransf, inverseTransf, eig_vals), or (None, None, None) on early failure.
        """
        img_cov = self.img_stats.get_cov()
        noise_cov = self.noise_stats.get_cov()

        # Ensure float64 and symmetry
        img_cov = np.asarray(img_cov, dtype=np.float32)
        noise_cov = np.asarray(noise_cov, dtype=np.float32)
        img_cov = 0.5 * (img_cov + img_cov.T)
        noise_cov = 0.5 * (noise_cov + noise_cov.T)

        # If covariance matrices are zero (very early), skip
        if not np.any(img_cov) or not np.any(noise_cov):
            return None, None, None

        # Try generalized eigenproblem, with adaptive regularization on img_cov if it is not SPD
        # Solve noise_cov v = lambda * img_cov v
        try:
            eig_vals, eig_vectors = eigh(noise_cov, img_cov)
        except np.linalg.LinAlgError:
            # adaptive regularization: eps proportional to trace(img_cov)/bands, grow if needed
            trace = np.trace(img_cov) if np.trace(img_cov) != 0 else 1.0
            eps = 1e-8 * (trace / max(1, self.bands))
            success = False
            for i in range(10):
                try:
                    img_cov_reg = img_cov + np.eye(self.bands) * eps
                    eig_vals, eig_vectors = eigh(noise_cov, img_cov_reg)
                    success = True
                    print(f"[Line_By_Line_MNF] Regularized img_cov with eps={eps:.3e} and succeeded.")
                    break
                except np.linalg.LinAlgError:
                    eps *= 10.0
            if not success:
                # fallback: try plain eigh on symmetrized generalized matrix via cholesky whitening
                try:
                    # Cholesky whitening safest fallback
                    L = np.linalg.cholesky(img_cov + np.eye(self.bands) * (eps * 10))
                    Linv = np.linalg.inv(L)
                    A = Linv.dot(noise_cov).dot(Linv.T)
                    u_vals, u_vecs = np.linalg.eigh(A)
                    # recover generalized eigenvectors
                    eig_vals = u_vals
                    eig_vectors = np.linalg.solve(L.T, u_vecs)
                except Exception as e:
                    print("[Line_By_Line_MNF] Eigen decomposition failed even after regularization:", e)
                    return None, None, None

        # Sort eigenvalues ascending (low -> high) like your previous code
        idx = np.argsort(eig_vals)
        eig_vals = eig_vals[idx]
        eig_vectors = eig_vectors[:, idx]

        # Save for later
        self.__final_eigen_values = eig_vals
        self.__final_eigen_vectors = eig_vectors

        # Compute inverse transform as matrix inverse (inverse of eigenvector matrix)
        try:
            inverse_transf = np.linalg.inv(eig_vectors)
        except np.linalg.LinAlgError:
            # if inv fails (singular), use pseudo-inverse as fallback
            inverse_transf = np.linalg.pinv(eig_vectors)
            print("[Line_By_Line_MNF] Warning: eig_vectors singular; used pseudo-inverse for inverse transform.")

        return eig_vectors, inverse_transf, eig_vals

    def mnf_linebyline_add_mean(self, mean, line):
        # line + outer(mean, ones)
        return line + np.outer(mean, self._one_samples)

    def mnf_linebyline_remove_mean(self, mean, line):
        # line - outer(mean, ones)
        return line - np.outer(mean, self._one_samples)

    def save_img(self, image=None, name=None):
        if image is None:
            image = self.image

        img_rio = image.transpose(1, 0, 2)

        meta = {
            "driver": "GTiff",
            "height": img_rio.shape[1],
            "width": img_rio.shape[2],
            "count": img_rio.shape[0],
            "dtype": img_rio.dtype,
            "transform": rasterio.transform.from_origin(0, 0, 1, 1),
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }

        out_name = name if name else self.basefilename + f"_line_by_line_modified_{self.mnf_config.noiseMatrixCalculation}_{self.numBandsInInv}.tif"
        with rasterio.open(out_name, "w", **meta) as dst:
            dst.write(img_rio)
        print(f"[Line_By_Line_MNF] saved {out_name}")
