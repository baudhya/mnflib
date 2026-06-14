"""
Benchmark all noise estimation methods on both datasets using both MNF modes.

Methods:  next_pixel, three_pixel, four_pixel, soft_diagonal
Modes:    MNF (whole-image), Line_By_Line_MNF
Datasets: Hysis_SW_500x500x125, reflectance

Output: results/benchmark_all.csv  (also printed as a table)
"""

import os
import time
import csv

import numpy as np

from mnflib.dataloader import GeotifImageLoader
from mnflib.mnf import MNF
from mnflib.mnf_linebyline import Line_By_Line_MNF
from mnflib.data_structures import TransformDirection, MNFConfig
from mnflib.scores import QualityMetricCalculator

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = {
    "hysis":        os.path.join(BASE, "dataset/Hysis_SW_500x500x125/HY1SW018901PS011803.tif"),
    "reflectance":  os.path.join(BASE, "dataset/reflectance/reflectance.tif"),
}

NOISE_METHODS = ["next_pixel", "three_pixel", "four_pixel", "soft_diagonal"]
BANDS_FRACTION = 0.2
OUTPUT_CSV = os.path.join(BASE, "results/benchmark_all.csv")


def run_one(image_path: str, noise_method: str, line_by_line: bool) -> dict:
    loader = GeotifImageLoader(image_path)
    original = loader.get_image()
    lines, bands, samples = original.shape

    cfg = MNFConfig(
        direction=TransformDirection.RUN_BOTH,
        basefilename=image_path,
        bands=bands,
        samples=samples,
        lines=lines,
        percentageOfBandsInInverse=BANDS_FRACTION,
        noiseMatrixCalculation=noise_method,
        profile=loader.profile,
    )

    mnf = Line_By_Line_MNF(original, cfg) if line_by_line else MNF(original, cfg)
    t0 = time.perf_counter()
    mnf.run()
    elapsed = time.perf_counter() - t0

    calc = QualityMetricCalculator(original, mnf.image)
    metrics = calc.calculate_all_metrics()
    snr_deltas = [s["snr_difference"] for s in calc.get_snr_stats()]

    return {
        "ssim":          metrics["ssim"],
        "psnr":          metrics["psnr"],
        "mse":           metrics["mse"],
        "mean_snr_gain": float(np.mean(snr_deltas)),
        "runtime_s":     elapsed,
    }


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    rows = []

    for ds_name, ds_path in DATASETS.items():
        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name}  ({ds_path})")
        print(f"{'='*70}")
        print(f"{'Mode':<16} {'Method':<16} {'SSIM':>8} {'PSNR':>8} {'MSE':>12} {'SNRgain':>9} {'Time(s)':>8}")
        print("-" * 82)

        for method in NOISE_METHODS:
            for line_by_line, mode_name in [(False, "MNF"), (True, "Line_By_Line")]:
                try:
                    m = run_one(ds_path, method, line_by_line)
                    print(
                        f"{mode_name:<16} {method:<16} "
                        f"{m['ssim']:>8.4f} {m['psnr']:>8.2f} "
                        f"{m['mse']:>12.4f} {m['mean_snr_gain']:>9.4f} "
                        f"{m['runtime_s']:>8.1f}"
                    )
                    rows.append({"dataset": ds_name, "mode": mode_name, "method": method, **m})
                except Exception as exc:
                    print(f"{mode_name:<16} {method:<16}  ERROR: {exc}")
                    rows.append({
                        "dataset": ds_name, "mode": mode_name, "method": method,
                        "ssim": None, "psnr": None, "mse": None,
                        "mean_snr_gain": None, "runtime_s": None,
                    })

    fieldnames = ["dataset", "mode", "method", "ssim", "psnr", "mse", "mean_snr_gain", "runtime_s"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
