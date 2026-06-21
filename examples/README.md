# Example Scripts

This directory contains standalone scripts that demonstrate how to use the `mnflib` package.

## Setup

Make sure `mnflib` is installed before running:

```bash
pip install -e .          # from project root
pip install -e ..         # from examples/ directory
```

Update any hardcoded `image_path` variables in the scripts to point to your data.

## Scripts

### visualize_band.py

Visualize individual spectral bands from a GeoTIFF.

```bash
python examples/visualize_band.py --image-path path/to/image.tif --band 1
python examples/visualize_band.py --folder path/to/folder --band 1  # batch mode
```

### plot_snr_per_band.py

Plot per-band SNR for different MNF noise estimation methods.

```bash
python examples/plot_snr_per_band.py
```

### plot_mnf_eigenvalues.py

Visualize and compare MNF eigenvalues across configurations.

```bash
python examples/plot_mnf_eigenvalues.py
```

### per_band_analysis.py

Compute per-band SSIM, PSNR, and SNR for all noise methods; writes CSVs and plots.

```bash
python examples/per_band_analysis.py
```

### benchmark_all.py

Benchmark all noise estimation methods across datasets and produce a summary CSV.

```bash
python examples/benchmark_all.py
```

### generate_results_table.py

Run MNF with multiple band-retention settings and generate CSV results tables.

```bash
python examples/generate_results_table.py
```

### test_hysis_data.py

End-to-end benchmark on HySIS and reflectance datasets; writes benchmark CSVs.

```bash
python examples/test_hysis_data.py
```
