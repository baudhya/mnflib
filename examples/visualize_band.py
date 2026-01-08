"""
Script to open a TIF file, select a band, and visualize it.

Usage:
    python visualize_band.py --image-path path/to/image.tif --band 1
    python visualize_band.py --folder path/to/folder --band 1  # Process all TIF files in folder
"""

import argparse
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


def load_tif(image_path: str) -> tuple:
    """
    Load a TIF file and return the data and metadata.
    
    Args:
        image_path: Path to the TIF file
        
    Returns:
        tuple: (data array, profile metadata, band count)
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {image_path}")
    
    with rasterio.open(path) as src:
        data = src.read()  # Shape: (bands, height, width)
        profile = src.profile
        band_count = src.count
        
    return data, profile, band_count


def visualize_band(data: np.ndarray, band_index: int, image_path: str = None,
                   title: str = None, cmap: str = 'viridis', save_path: str = None,
                   output_dir: str = None):
    """
    Visualize a specific band from the image data.
    
    Args:
        data: Image data array with shape (bands, height, width)
        band_index: 1-based band index to visualize
        image_path: Original image path (used for auto-generating save filename)
        title: Optional title for the plot
        cmap: Colormap to use for visualization
        save_path: Optional path to save the figure (auto-generated if not provided)
        output_dir: Optional output directory for saving visualizations
    """
    # Convert to 0-based index
    idx = band_index - 1
    
    if idx < 0 or idx >= data.shape[0]:
        raise ValueError(f"Band index {band_index} out of range. Valid range: 1-{data.shape[0]}")
    
    band_data = data[idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Handle potential NaN or inf values
    valid_data = np.ma.masked_invalid(band_data)
    
    # Display the band
    im = ax.imshow(valid_data, cmap=cmap)
    
    # Set title only if explicitly provided
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Column (pixels)', fontsize=11)
    ax.set_ylabel('Row (pixels)', fontsize=11)
    
    # Add statistics annotation
    stats_text = (
        f"Min: {np.nanmin(band_data):.4f}\n"
        f"Max: {np.nanmax(band_data):.4f}\n"
        f"Mean: {np.nanmean(band_data):.4f}\n"
        f"Std: {np.nanstd(band_data):.4f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Auto-generate save path using image name + band number
    if save_path is None:
        if image_path:
            base_name = Path(image_path).stem  # Get filename without extension
            filename = f"{base_name}_band_{band_index}.png"
        else:
            filename = f"band_{band_index}_visualization.png"
        
        # Use output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, filename)
        else:
            save_path = filename
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    # Try to show interactively, but don't fail if not possible
    try:
        if plt.get_backend() != 'agg':
            plt.show()
    except Exception:
        pass
    
    plt.close(fig)


def print_band_info(data: np.ndarray, profile: dict, band_count: int, image_name: str = None):
    """Print information about the TIF file."""
    print("\n" + "=" * 60)
    if image_name:
        print(f"TIF FILE: {image_name}")
    else:
        print("TIF FILE INFORMATION")
    print("=" * 60)
    print(f"  Number of bands: {band_count}")
    print(f"  Image dimensions: {data.shape[2]} x {data.shape[1]} (width x height)")
    print(f"  Data type: {data.dtype}")
    print(f"  CRS: {profile.get('crs', 'Not specified')}")
    print("=" * 60 + "\n")


def interactive_band_selection(band_count: int) -> int:
    """Interactively select a band from user input."""
    while True:
        try:
            band = int(input(f"Enter band number (1-{band_count}): "))
            if 1 <= band <= band_count:
                return band
            else:
                print(f"Please enter a number between 1 and {band_count}")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit(0)


def get_tif_files(folder_path: str) -> list:
    """Get all TIF files in a folder."""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    tif_files = list(folder.glob("*.tif")) + list(folder.glob("*.tiff"))
    return sorted(tif_files)


def process_folder(folder_path: str, band: int, cmap: str = 'viridis', 
                   output_dir: str = None):
    """
    Process all TIF files in a folder.
    
    Args:
        folder_path: Path to folder containing TIF files
        band: Band number to visualize
        cmap: Colormap to use
        output_dir: Output directory for visualizations
    """
    tif_files = get_tif_files(folder_path)
    
    if not tif_files:
        print(f"No TIF files found in: {folder_path}")
        return
    
    print(f"\nFound {len(tif_files)} TIF files in: {folder_path}")
    print("-" * 60)
    
    # Use output directory in the folder if not specified
    if output_dir is None:
        output_dir = os.path.join(folder_path, "band_visualizations")
    
    for i, tif_file in enumerate(tif_files, 1):
        print(f"\n[{i}/{len(tif_files)}] Processing: {tif_file.name}")
        
        try:
            data, profile, band_count = load_tif(str(tif_file))
            
            # Check if band is valid for this file
            if band > band_count:
                print(f"  Warning: Band {band} not available (file has {band_count} bands). Skipping.")
                continue
            
            visualize_band(
                data=data,
                band_index=band,
                image_path=str(tif_file),
                cmap=cmap,
                output_dir=output_dir
            )
            
        except Exception as e:
            print(f"  Error processing {tif_file.name}: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"Batch processing complete. Visualizations saved to: {output_dir}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Open a TIF file, select a band, and visualize it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file:
  python visualize_band.py --image-path image.tif --band 1
  python visualize_band.py --image-path image.tif --band 5 --cmap gray
  
  # Process all TIF files in a folder:
  python visualize_band.py --folder /path/to/folder --band 1
  python visualize_band.py --folder /path/to/folder --band 5 --output-dir /path/to/output
        """
    )
    
    # Mutually exclusive group for single file vs folder
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image-path", "-i",
        type=str,
        help="Path to a single TIF file"
    )
    input_group.add_argument(
        "--folder", "-f",
        type=str,
        help="Path to folder containing TIF files (processes all TIF files)"
    )
    
    parser.add_argument(
        "--band", "-b",
        type=int,
        default=None,
        help="Band number to visualize (1-indexed). Required for folder mode."
    )
    
    parser.add_argument(
        "--cmap", "-c",
        type=str,
        default="viridis",
        help="Matplotlib colormap to use (default: viridis). Options: gray, jet, plasma, magma, etc."
    )
    
    parser.add_argument(
        "--save", "-s",
        type=str,
        default=None,
        help="Optional path to save the figure (single file mode only)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for visualizations (folder mode)"
    )
    
    parser.add_argument(
        "--title", "-t",
        type=str,
        default=None,
        help="Optional title for the visualization (single file mode only)"
    )
    
    args = parser.parse_args()
    
    # Folder mode
    if args.folder:
        if args.band is None:
            parser.error("--band is required when using --folder mode")
        
        process_folder(
            folder_path=args.folder,
            band=args.band,
            cmap=args.cmap,
            output_dir=args.output_dir
        )
    
    # Single file mode
    else:
        print(f"Loading TIF file: {args.image_path}")
        data, profile, band_count = load_tif(args.image_path)
        
        # Print file information
        print_band_info(data, profile, band_count, Path(args.image_path).name)
        
        # Select band
        if args.band is not None:
            band = args.band
        else:
            band = interactive_band_selection(band_count)
        
        print(f"Visualizing band {band}...")
        
        # Visualize the band
        visualize_band(
            data=data,
            band_index=band,
            image_path=args.image_path,
            title=args.title,
            cmap=args.cmap,
            save_path=args.save
        )


if __name__ == "__main__":
    main()
