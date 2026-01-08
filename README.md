# mnflib

**MNF (Minimum Noise Fraction) Transform Library for Hyperspectral Image Processing**

A Python library for applying MNF transforms to hyperspectral images for noise reduction and dimensionality reduction.

## Features

- **MNF Transform**: Full-image Minimum Noise Fraction transformation
- **Line-by-Line MNF**: Memory-efficient line-by-line processing for large images
- **Multiple Noise Estimation Methods**: Support for `next_pixel` and `three_pixel` noise estimation
- **GeoTIFF Support**: Read and write GeoTIFF files with preserved geospatial metadata
- **Quality Metrics**: SNR, SSIM, PSNR, and variance calculations

## Installation

### From PyPI (when published)

```bash
pip install mnflib
```

### From Source

```bash
git clone https://github.com/yourusername/mnflib.git
cd mnflib
pip install -e .
```

## Quick Start

### Python API

```python
from mnflib import MNF, GeotifImageLoader, MNFConfig, TransformDirection

# Load image
loader = GeotifImageLoader("path/to/image.tif")
image = loader.get_image()

lines, bands, samples = image.shape

# Configure MNF
config = MNFConfig(
    direction=TransformDirection.RUN_BOTH,
    basefilename="path/to/image.tif",
    bands=bands,
    samples=samples,
    lines=lines,
    percentageOfBandsInInverse=0.1,  # Use top 10% of bands
    noiseMatrixCalculation="next_pixel"
)

# Run MNF
mnf = MNF(image, config)
result = mnf.run()

# Access processed image
processed_image = mnf.image
```

### Command Line Interface

```bash
# Run MNF transform
mnf --image-path path/to/image.tif --noise-matrix next_pixel --inverse-bands-percentage 0.1

# With line-by-line processing
mnf --image-path path/to/image.tif --line-by-line --noise-matrix three_pixel
```

## API Reference

### Classes

- **`MNF`**: Main MNF transform class for whole-image processing
- **`Line_By_Line_MNF`**: Memory-efficient line-by-line MNF processing
- **`GeotifImageLoader`**: Load GeoTIFF images
- **`MNFConfig`**: Configuration dataclass for MNF parameters
- **`TransformDirection`**: Enum for transform direction (FORWARD, INVERSE, BOTH)

### Configuration Options

| Parameter | Description |
|-----------|-------------|
| `direction` | Transform direction: `RUN_FORWARD`, `RUN_INVERSE`, or `RUN_BOTH` |
| `noiseMatrixCalculation` | Noise estimation method: `"next_pixel"` or `"three_pixel"` |
| `percentageOfBandsInInverse` | Fraction of bands to use in inverse transform (0.0-1.0) |

## Examples

See the `examples/` directory for usage examples:

- `visualize_band.py` - Visualize individual bands from TIF files
- `plot_snr_per_band.py` - Plot SNR across bands for different configurations
- `plot_mnf_eigenvalues.py` - Visualize MNF eigenvalues

## License

MIT License - see [LICENSE](LICENSE) for details.
