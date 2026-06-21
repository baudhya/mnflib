#!/usr/bin/env python3
"""
Test MNF on the HySIS hyperspectral datasets.

Runs both whole-image and line-by-line MNF across noise estimation methods,
prints a results table, and writes a CSV summary to results/hysis_test/.

Usage:
    python examples/test_hysis_data.py [--quick]

    --quick  Skip the large reflectance image (2199×1280×40) to save time.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

# Allow running from any working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mnflib import (
    MNF,
    Line_By_Line_MNF,
    GeotifImageLoader,
    MNFConfig,
    QualityMetricCalculator,
    TransformDirection,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

IMAGES = {
    "hysis": os.path.join(
        _ROOT, "dataset", "Hysis_SW_500x500x125", "HY1SW018901PS011803.tif"
    ),
    "reflectance": os.path.join(_ROOT, "dataset", "reflectance", "reflectance.tif"),
}

RESULTS_DIR = os.path.join(_ROOT, "results", "hysis_test")

# Configurations to benchmark: (noise_method, pct_bands_in_inverse)
NOISE_METHODS = ["next_pixel", "three_pixel"]
PCT_BANDS = 0.2   # keep top 20 % of bands in inverse transform

# Note on line-by-line quality: the incremental covariance estimate is
# unreliable until many lines have been seen.  For wide-band images (e.g.
# 125 bands) reconstruction quality on early lines is poor and drags down
# the global metrics.  Whole-image MNF will always score better here.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(char="─", width=80):
    print(char * width)


def _header(title: str):
    _sep("═")
    print(f"  {title}")
    _sep("═")


def _load_image(name: str, path: str):
    print(f"\nLoading {name} ...")
    loader = GeotifImageLoader(path)
    img = loader.get_image()
    lines, bands, samples = img.shape
    print(f"  Shape : {lines} lines × {bands} bands × {samples} samples")
    print(f"  dtype : {img.dtype}")
    print(f"  Range : [{img.min():.4f}, {img.max():.4f}]")
    return img, loader.profile


def _run_mnf(
    image: np.ndarray,
    name: str,
    method: str,
    mode: str,
    pct: float,
    image_path: str,
    profile: dict,
) -> dict:
    lines, bands, samples = image.shape
    cfg = MNFConfig(
        direction=TransformDirection.RUN_BOTH,
        image_path=image_path,
        bands=bands,
        samples=samples,
        lines=lines,
        percentageOfBandsInInverse=pct,
        noiseMatrixCalculation=method,
        profile=profile,
    )

    t0 = time.perf_counter()
    if mode == "whole_image":
        transformer = MNF(image, cfg)
    else:
        transformer = Line_By_Line_MNF(image, cfg)
    result = transformer.run()
    elapsed = time.perf_counter() - t0

    calc = QualityMetricCalculator(image, transformer.image)
    metrics = calc.calculate_all_metrics()
    snr_stats = calc.get_snr_stats()
    mean_snr_orig = float(np.mean([s["snr_original"]  for s in snr_stats]))
    mean_snr_proc = float(np.mean([s["snr_processed"] for s in snr_stats]))

    return {
        "image":        name,
        "mode":         mode,
        "noise_method": method,
        "pct_bands":    pct,
        "time_s":       round(elapsed, 2),
        "ssim":         round(metrics["ssim"], 6),
        "mse":          round(metrics["mse"], 6),
        "psnr_db":      round(metrics["psnr"], 4),
        "var_before":   round(metrics["variance_original"], 6),
        "var_after":    round(metrics["variance_processed"], 6),
        "mean_snr_orig": round(mean_snr_orig, 4),
        "mean_snr_proc": round(mean_snr_proc, 4),
        "top5_eigenvalues": (
            [round(float(v), 6) for v in result.eigenvalues[:5]]
            if result.eigenvalues is not None
            else []
        ),
    }


def _print_row(row: dict):
    _sep()
    print(
        f"  Image       : {row['image']}\n"
        f"  Mode        : {row['mode']}\n"
        f"  Noise method: {row['noise_method']}\n"
        f"  Bands kept  : {int(row['pct_bands'] * 100)} %\n"
        f"  Time        : {row['time_s']:.2f} s\n"
        f"  SSIM        : {row['ssim']:.6f}\n"
        f"  MSE         : {row['mse']:.6f}\n"
        f"  PSNR        : {row['psnr_db']:.4f} dB\n"
        f"  Var before  : {row['var_before']:.6f}\n"
        f"  Var after   : {row['var_after']:.6f}\n"
        f"  Mean SNR before: {row['mean_snr_orig']:.4f}\n"
        f"  Mean SNR after : {row['mean_snr_proc']:.4f}\n"
        f"  Top-5 eigenvalues: {row['top5_eigenvalues']}"
    )


def _save_csv(rows: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    fieldnames = [k for k in rows[0] if k != "top5_eigenvalues"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v for k, v in row.items() if k != "top5_eigenvalues"})
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip the large reflectance image to save time"
    )
    args = parser.parse_args()

    images_to_test = (
        {k: v for k, v in IMAGES.items() if k == "hysis"}
        if args.quick
        else IMAGES
    )

    # Validate paths upfront.
    missing = [p for p in images_to_test.values() if not os.path.exists(p)]
    if missing:
        print("ERROR: the following dataset files were not found:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    all_rows = []

    for img_name, img_path in images_to_test.items():
        _header(f"Dataset: {img_name}")
        image, profile = _load_image(img_name, img_path)

        for method in NOISE_METHODS:
            for mode in ["whole_image", "line_by_line"]:
                label = f"{img_name} / {mode} / {method}"
                print(f"\nRunning: {label} ...")
                try:
                    row = _run_mnf(
                        image=image,
                        name=img_name,
                        method=method,
                        mode=mode,
                        pct=PCT_BANDS,
                        image_path=img_path,
                        profile=profile,
                    )
                    _print_row(row)
                    all_rows.append(row)
                except Exception as exc:
                    print(f"  FAILED: {exc}")

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    if all_rows:
        _header("SUMMARY")
        col_w = [18, 14, 14, 8, 8, 8, 8, 14, 14]
        headers = ["image", "mode", "noise_method", "time_s", "ssim",
                   "psnr_db", "mse", "mean_snr_orig", "mean_snr_proc"]
        fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*headers))
        _sep()
        for row in all_rows:
            print(fmt.format(
                row["image"], row["mode"], row["noise_method"],
                f"{row['time_s']:.2f}s",
                f"{row['ssim']:.4f}",
                f"{row['psnr_db']:.2f}",
                f"{row['mse']:.4f}",
                f"{row['mean_snr_orig']:.4f}",
                f"{row['mean_snr_proc']:.4f}",
            ))

        csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
        _save_csv(all_rows, csv_path)

    _sep("═")
    print("  Done.")
    _sep("═")


if __name__ == "__main__":
    main()
