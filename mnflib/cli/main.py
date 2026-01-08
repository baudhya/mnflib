"""Command-line interface for mnflib."""

import os
import argparse

from mnflib.dataloader import GeotifImageLoader
from mnflib.mnf import MNF
from mnflib.mnf_linebyline import Line_By_Line_MNF
from mnflib.data_structures import TransformDirection, MNFConfig
from mnflib.utils import noise_added_image
from mnflib.scores import QualityMetricCalculator


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MNF (Minimum Noise Fraction) Transform for Hyperspectral Image Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mnf --image-path image.tif
  mnf --image-path image.tif --noise-matrix three_pixel
  mnf --image-path image.tif --line-by-line --inverse-bands-percentage 0.1
        """
    )
    
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to the GeoTIFF image file"
    )
    
    parser.add_argument(
        "--transform-direction",
        type=str,
        choices=["forward", "inverse", "both"],
        default="both",
        help="Transform direction: forward, inverse, or both (default: both)"
    )
    
    parser.add_argument(
        "--inverse-bands-percentage",
        type=float,
        default=0.2,
        help="Percentage of bands to use in inverse transform, 0.0-1.0 (default: 0.2)"
    )
    
    parser.add_argument(
        "--add-noise",
        type=float,
        default=0,
        help="Standard deviation for Gaussian noise. 0 = no noise (default: 0)"
    )
    
    parser.add_argument(
        "--line-by-line",
        action="store_true",
        help="Run line-by-line MNF processing instead of whole image"
    )

    parser.add_argument(
        "--noise-matrix",
        choices=["next_pixel", "four_pixel", "three_pixel"],
        default="next_pixel",
        help="Noise matrix calculation method (default: next_pixel)"
    )
    
    parser.add_argument(
        "--results",
        action="store_true",
        help="Print all quality metrics (SSIM, MSE, PSNR, SNR, variance per band)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the MNF CLI."""
    args = parse_arguments()
    
    # Validate image path
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return 1
    
    # Map transform direction string to enum
    trans_dir_map = {
        "forward": TransformDirection.RUN_FORWARD,
        "inverse": TransformDirection.RUN_INVERSE,
        "both": TransformDirection.RUN_BOTH
    }
    trans_dir = trans_dir_map[args.transform_direction]
    
    # Load image
    print(f"Loading image: {args.image_path}")
    image_loader = GeotifImageLoader(args.image_path)
    original_image = image_loader.get_image()
    
    # Add noise if specified
    if args.add_noise > 0:
        print(f"Adding Gaussian noise with std_dev={args.add_noise}")
        noisy_image = noise_added_image(original_image, mean=0, std_dev=args.add_noise)
    else:
        noisy_image = original_image
    
    lines, bands, samples = original_image.shape
    print(f"Image shape: {lines} lines x {bands} bands x {samples} samples")
    
    mnf_config = MNFConfig(
        trans_dir,
        args.image_path,
        bands,
        samples,
        lines,
        args.inverse_bands_percentage,
        args.noise_matrix
    )
    
    print("-" * 50)
    print(f"Transform direction: {args.transform_direction}")
    print(f"Noise matrix method: {args.noise_matrix}")
    print(f"Inverse bands percentage: {args.inverse_bands_percentage}")
    print(f"Line-by-line mode: {args.line_by_line}")
    print("-" * 50)
    
    if args.line_by_line:
        print("Running Line-by-Line MNF...")
        mnf = Line_By_Line_MNF(noisy_image, mnf_config)
    else:
        print("Running Whole Image MNF...")
        mnf = MNF(noisy_image, mnf_config)
    
    result = mnf.run()
    
    if result.eigenvalues is not None:
        print("\nTop 10 Eigenvalues:")
        for i, ev in enumerate(result.eigenvalues[:10]):
            print(f"  {i+1}: {ev:.6f}")
    
    # Print quality metrics if --results flag is set
    if args.results:
        print("\n" + "=" * 50)
        print("QUALITY METRICS")
        print("=" * 50)
        
        calculator = QualityMetricCalculator(noisy_image, mnf.image)
        
        # Global metrics
        metrics = calculator.calculate_all_metrics()
        print("\n--- Global Metrics ---")
        print(f"SSIM Score: {metrics['ssim_original']:.6f}")
        print(f"Mean Square Error: {metrics['mse']:.6f}")
        print(f"PSNR Score: {metrics['psnr']:.6f}")
        print(f"Variance Before MNF: {metrics['variance_original']:.6f}")
        print(f"Variance After MNF: {metrics['variance_processed']:.6f}")
        
        # SNR stats per band
        print("\n--- SNR per Band ---")
        snr_stats = calculator.get_snr_stats()
        print(f"{'Band':>6} | {'Original':>12} | {'Processed':>12} | {'Difference':>12}")
        print("-" * 50)
        for stat in snr_stats:
            print(f"{stat['band']:>6} | {stat['snr_original']:>12.4f} | {stat['snr_processed']:>12.4f} | {stat['snr_difference']:>12.4f}")
        
        # SSIM per band
        print("\n--- SSIM per Band ---")
        band_ssim = calculator.get_band_ssim()
        for item in band_ssim:
            print(f"Band {item['band']:>3}: {item['ssim']:.4f}")
        
        # Variance per band
        print("\n--- Variance per Band ---")
        variance = calculator.get_variance_stats()
        for bnd, val in variance.items():
            print(f"Band {bnd:>3}: {val:.6e}")
        
        print("=" * 50)
    
    print("\nMNF transform completed successfully.")
    return 0


if __name__ == "__main__":
    exit(main())
