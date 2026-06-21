# mnflib

**MNF (Minimum Noise Fraction) Transform Library for Hyperspectral Image Processing**

A Python library for applying MNF transforms to hyperspectral images for noise reduction and dimensionality reduction. Implements the Three-Pixel Noise Estimation (TPNE) and Four-Pixel Noise Estimation (4PNE) methods proposed in the accompanying research paper.

## Features

- **Global MNF**: Whole-image Minimum Noise Fraction transformation
- **Line-by-Line MNF**: Memory-efficient streaming processing for large images
- **Four Noise Estimation Methods**: `next_pixel` (NPNE), `three_pixel` (TPNE), `four_pixel` (4PNE), `soft_diagonal`
- **GeoTIFF Support**: Read and write GeoTIFF files with full geospatial metadata preservation
- **Quality Metrics**: Per-band SNR, SSIM, PSNR, MSE, and variance calculations
- **Adaptive Regularisation**: Tikhonov regularisation and Cholesky whitening fallback for ill-conditioned covariance matrices
- **CLI**: Command-line interface for quick processing without writing code

## Installation

```bash
git clone https://github.com/baudhya/mnflib.git
cd mnflib
pip install -e .          # basic install
pip install -e ".[dev]"   # include dev dependencies (pytest, build, twine)
```

## Quick Start

### Python API

```python
from mnflib import MNF, GeotifImageLoader, MNFConfig, TransformDirection

# Load image  (returned as (lines, bands, samples) float32)
loader = GeotifImageLoader("path/to/image.tif")
image  = loader.get_image()
lines, bands, samples = image.shape

# Configure MNF with TPNE noise estimation
config = MNFConfig(
    direction=TransformDirection.RUN_BOTH,
    image_path="path/to/image.tif",
    bands=bands,
    samples=samples,
    lines=lines,
    percentageOfBandsInInverse=0.2,        # keep top 20% of bands
    noiseMatrixCalculation="three_pixel",  # TPNE
    profile=loader.profile,                # preserve geospatial metadata
)

# Run MNF
transformer = MNF(image, config)
result = transformer.run()

# Access denoised image
processed = transformer.image
print(result.eigenvalues[:5])   # top-5 eigenvalues (noise fractions)
```

### Line-by-Line (streaming) mode

```python
from mnflib import Line_By_Line_MNF

transformer = Line_By_Line_MNF(image, config)
transformer.run()
```

### Command Line Interface

```bash
# Basic MNF with default settings
mnf --image-path path/to/image.tif

# TPNE noise estimation, keep top 20% bands
mnf --image-path image.tif --noise-matrix three_pixel --inverse-bands-percentage 0.2

# Line-by-line mode with 4PNE
mnf --image-path image.tif --line-by-line --noise-matrix four_pixel

# Forward transform only
mnf --image-path image.tif --transform-direction forward

# Print quality metrics after processing
mnf --image-path image.tif --results
```

## Noise Estimation Methods

| Method | Key | Description |
|--------|-----|-------------|
| Next-Pixel (NPNE) | `next_pixel` | Horizontal pixel difference only |
| Three-Pixel (TPNE) | `three_pixel` | Average of horizontal, vertical and diagonal differences |
| Four-Pixel (4PNE) | `four_pixel` | Average of horizontal, vertical, below-horizontal and diagonal differences |
| Soft Diagonal | `soft_diagonal` | MAD-thresholded diagonal difference; suppresses signal contamination |

TPNE and 4PNE capture noise in multiple spatial directions, producing a more stable noise covariance matrix. Experimental results on a 40-band reflectance dataset show TPNE outperforms NPNE on SNR gain in **35 of 40 bands**, with all four methods improving 100% of spectral bands.

## API Reference

### Classes

| Class | Description |
|-------|-------------|
| `MNF` | Whole-image MNF transform |
| `Line_By_Line_MNF` | Streaming line-by-line MNF |
| `GeotifImageLoader` | Load GeoTIFF images into `(lines, bands, samples)` arrays |
| `MNFConfig` | Configuration dataclass for all run parameters |
| `TransformDirection` | Enum: `RUN_FORWARD`, `RUN_INVERSE`, `RUN_BOTH` |
| `QualityMetricCalculator` | Compute SNR, SSIM, PSNR, MSE between original and processed images |

### MNFConfig Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `direction` | `TransformDirection` | Which transform(s) to run |
| `image_path` | `str` | Path to the input GeoTIFF; output dirs are written alongside it |
| `bands` / `samples` / `lines` | `int` | Image dimensions (read from `loader.get_image().shape`) |
| `noiseMatrixCalculation` | `str` | `"next_pixel"`, `"three_pixel"`, `"four_pixel"`, or `"soft_diagonal"` |
| `percentageOfBandsInInverse` | `float` | Fraction of highest-SNR bands to keep in reconstruction (0, 1] |
| `profile` | `dict` | Rasterio profile from loader — preserves CRS and geotransform |

## Output Layout

All outputs are written next to the input image file:

```
output_images/   — denoised GeoTIFF (LZW-compressed, tiled 256×256)
eigen_data/      — eigenvalue .dat files
stats_data/      — covariance and mean .pkl files (used by RUN_INVERSE)
```

## Examples

```bash
python examples/benchmark_all.py                               # benchmark all noise methods across configs
python examples/generate_results_table.py                      # produce CSV results tables
python examples/plot_snr_per_band.py                           # SNR comparison plots
python examples/plot_mnf_eigenvalues.py                        # eigenvalue visualisation
python examples/per_band_analysis.py                           # per-band SSIM/PSNR/SNR comparison
python examples/test_hysis_data.py                             # benchmark on HySIS and reflectance datasets
python examples/visualize_band.py --image-path image.tif --band 1
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Image Array Convention

**Throughout the library, arrays are stored as `(lines, bands, samples)`**, not the rasterio default of `(bands, lines, samples)`. `GeotifImageLoader` transposes on load automatically.

## Research Paper

This library accompanies the paper:

> **Minimum Noise Fraction with Multi-Directional Noise Estimation for Denoising Hyperspectral Imagery**
> Siddharth Baudh, Kamal Deep
> Defence Geoinformatics Research Establishment, Chandigarh, India

The paper introduces the TPNE and 4PNE noise estimation methods and evaluates them against the conventional NPNE approach on real hyperspectral datasets. See `MNF_Paper_Final.docx` for the full paper.

## License

MIT License — see [LICENSE](LICENSE) for details.
