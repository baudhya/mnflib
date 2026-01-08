# Example Scripts

This directory contains standalone scripts that demonstrate how to use the `mnflib` package.

## Scripts

### visualize_band.py

Visualize individual bands from TIF files.

```bash
# Single file mode
python visualize_band.py --image-path path/to/image.tif --band 1

# Batch mode - process all TIF files in a folder
python visualize_band.py --folder path/to/folder --band 1
```

### plot_snr_per_band.py

Plot SNR (Signal-to-Noise Ratio) across bands for different MNF configurations.

```bash
python plot_snr_per_band.py
```

Note: Update the `image_path` variable in the script before running.

### plot_mnf_eigenvalues.py

Visualize and compare MNF eigenvalues across different configurations.

```bash
python plot_mnf_eigenvalues.py
```

### variance_table_generator.py

Parse MNF output results and generate variance tables in CSV and Markdown formats.

```bash
python variance_table_generator.py
```

### run.py

Batch runner script for executing MNF transforms with multiple configurations.

```bash
python run.py
```

### argument_parser.py

Argument parser utility used by the main CLI.

## Usage Notes

1. These scripts are designed to be run from the project root directory or the `examples/` directory.

2. Make sure `mnflib` is installed before running:
   ```bash
   pip install -e ..  # If running from examples/
   # or
   pip install -e .   # If running from project root
   ```

3. Update hardcoded paths in scripts as needed for your environment.
