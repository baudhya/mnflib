"""Command-line interface for mnflib."""

import os
import argparse

from mnflib.dataloader import GeotifImageLoader
from mnflib.mnf import MNF
from mnflib.mnf_linebyline import Line_By_Line_MNF
from mnflib.data_structures import TransformDirection, MNFConfig
from mnflib.utils import noise_added_image
from mnflib.scores import QualityMetricCalculator

_DIRECTION_MAP = {
    "forward": TransformDirection.RUN_FORWARD,
    "inverse": TransformDirection.RUN_INVERSE,
    "both":    TransformDirection.RUN_BOTH,
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="MNF (Minimum Noise Fraction) Transform for Hyperspectral Image Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mnf --image-path image.tif
  mnf --image-path image.tif --noise-matrix three_pixel
  mnf --image-path image.tif --line-by-line --inverse-bands-percentage 0.1
        """,
    )
    parser.add_argument("--image-path", type=str, required=True,
                        help="Path to the GeoTIFF image file")
    parser.add_argument("--transform-direction", type=str,
                        choices=list(_DIRECTION_MAP), default="both",
                        help="Transform direction (default: both)")
    parser.add_argument("--inverse-bands-percentage", type=float, default=0.2,
                        help="Fraction of bands kept in inverse transform, 0-1 (default: 0.2)")
    parser.add_argument("--add-noise", type=float, default=0,
                        help="Gaussian noise std-dev to add before processing (default: 0)")
    parser.add_argument("--line-by-line", action="store_true",
                        help="Use line-by-line MNF instead of whole-image MNF")
    parser.add_argument("--noise-matrix",
                        choices=["next_pixel", "three_pixel", "four_pixel", "soft_diagonal"],
                        default="next_pixel",
                        help="Noise estimation method (default: next_pixel)")
    parser.add_argument("--results", action="store_true",
                        help="Print quality metrics after processing")
    return parser.parse_args()


def main():
    args = parse_arguments()

    if not os.path.exists(args.image_path):
        print(f"Error: image file not found: {args.image_path}")
        return 1

    if not (0.0 < args.inverse_bands_percentage <= 1.0):
        print(f"Error: --inverse-bands-percentage must be in (0, 1], "
              f"got {args.inverse_bands_percentage}")
        return 1

    print(f"Loading image: {args.image_path}")
    loader = GeotifImageLoader(args.image_path)
    original_image = loader.get_image()
    lines, bands, samples = original_image.shape
    print(f"Image shape: {lines} lines × {bands} bands × {samples} samples")

    input_image = (
        noise_added_image(original_image, mean=0, std_dev=args.add_noise)
        if args.add_noise > 0
        else original_image
    )

    mnf_config = MNFConfig(
        direction=_DIRECTION_MAP[args.transform_direction],
        basefilename=args.image_path,
        bands=bands,
        samples=samples,
        lines=lines,
        percentageOfBandsInInverse=args.inverse_bands_percentage,
        noiseMatrixCalculation=args.noise_matrix,
        profile=loader.profile,
    )

    print("-" * 50)
    print(f"Direction:              {args.transform_direction}")
    print(f"Noise matrix method:    {args.noise_matrix}")
    print(f"Inverse bands fraction: {args.inverse_bands_percentage}")
    print(f"Line-by-line mode:      {args.line_by_line}")
    print("-" * 50)

    mnf = (
        Line_By_Line_MNF(input_image, mnf_config)
        if args.line_by_line
        else MNF(input_image, mnf_config)
    )
    result = mnf.run()

    if result.eigenvalues is not None:
        print("\nTop 10 eigenvalues:")
        for i, ev in enumerate(result.eigenvalues[:10]):
            print(f"  {i + 1:>2}: {ev:.6f}")

    if args.results:
        print("\n" + "=" * 50)
        print("QUALITY METRICS")
        print("=" * 50)
        calc = QualityMetricCalculator(input_image, mnf.image)
        metrics = calc.calculate_all_metrics()
        print(f"\nSSIM:               {metrics['ssim']:.6f}")
        print(f"MSE:                {metrics['mse']:.6f}")
        print(f"PSNR:               {metrics['psnr']:.6f}")
        print(f"Variance (before):  {metrics['variance_original']:.6f}")
        print(f"Variance (after):   {metrics['variance_processed']:.6f}")

        print("\n--- SNR per band ---")
        print(f"{'Band':>5} | {'Original':>12} | {'Processed':>12} | {'Delta':>10}")
        print("-" * 46)
        for s in calc.get_snr_stats():
            print(
                f"{s['band']:>5} | {s['snr_original']:>12.4f} | "
                f"{s['snr_processed']:>12.4f} | {s['snr_difference']:>10.4f}"
            )
        print("=" * 50)

    print("\nMNF transform completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
