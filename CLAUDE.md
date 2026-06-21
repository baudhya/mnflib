# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Installation

```bash
pip install -e .          # Install in editable mode (run from project root)
pip install -e ".[dev]"   # Include dev dependencies (pytest, build, twine)
```

## Running the CLI

```bash
# Basic MNF transform (whole-image)
mnf --image-path path/to/image.tif

# Common options
mnf --image-path image.tif --noise-matrix three_pixel --inverse-bands-percentage 0.1
mnf --image-path image.tif --line-by-line --noise-matrix next_pixel
mnf --image-path image.tif --results    # also print quality metrics

# Transform direction: forward, inverse, both (default: both)
mnf --image-path image.tif --transform-direction forward
```

## Running Example Scripts

```bash
python examples/plot_snr_per_band.py      # SNR comparison plots
python examples/plot_mnf_eigenvalues.py   # eigenvalue visualization
python examples/visualize_band.py --image-path image.tif --band 1
```

## Architecture

### Image Array Convention

**Throughout the codebase, arrays are stored as `(lines, bands, samples)`**, NOT the standard rasterio convention of `(bands, lines, samples)`. `GeotifImageLoader` transposes on load (`np.transpose(..., (1, 0, 2))`). Any new code must follow this convention.

### Two Processing Modes

**`MNF` (whole-image)** — `mnflib/mnf.py`
- Loads the entire image into memory as float32.
- Computes image and noise covariance matrices over all pixels at once (`ImageStatisticsFull`).
- Uses `scipy.linalg.eigh(noise_cov, img_cov)` to solve the generalized eigenvalue problem.
- Applies the forward/inverse transform line-by-line using BLAS GEMM (`F_T @ line`).

**`Line_By_Line_MNF`** — `mnflib/mnf_linebyline.py`
- Memory-efficient streaming variant: updates covariance incrementally as each line is processed.
- Uses `ImageStatistics.update_with_line()` (Welford-style accumulation via BLAS calls).
- Recomputes eigenvectors every line; handles near-singular covariance with adaptive Tikhonov regularization and Cholesky whitening fallback.
- On the first few lines the covariance matrix is zero, so `_compute_transform_matrices` returns `None` and the original line is passed through unchanged.

### MNF Algorithm Flow

1. Estimate noise using one of four methods (see below).
2. Compute image covariance `Σ_img` and noise covariance `Σ_noise`.
3. Solve `Σ_noise · v = λ · Σ_img · v` — eigenvalues are noise-fraction per band.
4. Sort eigenvalues **ascending** (lowest noise-fraction = highest SNR first).
5. Forward transform: `F_T @ (line - mean)`.
6. Reconstruction: apply diagonal selector matrix `R` (keeps first `numBandsInInv` components), then `F_inv_T @ R @ line + mean`.

### Noise Estimation Methods (`mnflib/utils.py`)

| Method | Function | Description |
|--------|----------|-------------|
| `next_pixel` | `calculate_noise_next_pixel` | Horizontal diff only; no previous line needed |
| `three_pixel` | `calculate_noise_three_pixel` | Average of horizontal, vertical, and diagonal diffs |
| `four_pixel` | `calculate_noise_four_pixel` | Average of four neighbors |
| `soft_diagonal` | `calculate_noise_soft_threshold_diagonal_diff` | MAD-thresholded diagonal diff |

`Line_By_Line_MNF` implements all four methods inline in `_estimate_noise()`, caching the previous line in `_cache_prev_line` for the multi-directional methods.

### Output Layout

Both `MNF` and `Line_By_Line_MNF` create three subdirectories next to the input image file:
- `output_images/` — reconstructed GeoTIFF (LZW-compressed, tiled 256×256)
- `eigen_data/` — eigenvalue `.dat` files
- `stats_data/` — serialized covariance and mean `.pkl` files

For `RUN_INVERSE`-only mode, `MNF` reads stats back from `stats_data/` via `_load_stats()`.

### Key Data Structures (`mnflib/data_structures.py`)

- `MNFConfig` — dataclass holding all run parameters; passed to both transform classes. Key fields: `image_path` (input GeoTIFF path; output dirs are created alongside it), `percentageOfBandsInInverse` (fraction of highest-SNR bands to keep), `noiseMatrixCalculation` (noise method string).
- `TransformDirection` — enum: `RUN_FORWARD`, `RUN_INVERSE`, `RUN_BOTH`.
- `MNFResult` — returned by `run()`; holds eigenvalues (noise fractions, ascending), eigenvectors, means, covariances.
- `ImageStatistics` — incremental stats (used by `Line_By_Line_MNF`); `get_cov()` divides by `n`.
- `ImageStatisticsFull` — whole-image stats (used by `MNF`); `get_cov()` returns `C` directly (already divided during `np.cov`).
- `ImageSubset` — optional spatial window passed to `GeotifImageLoader` to load a sub-region.
