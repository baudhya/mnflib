#!/usr/bin/env python3
"""
Per-band SSIM, PSNR, and SNR analysis across MNF noise estimation methods.

For each spectral band and each method (next_pixel / three_pixel / four_pixel):
  - SNR before and after MNF, SNR gain
  - SSIM between original and reconstructed band
  - PSNR between original and reconstructed band

Summary counts how many bands are enhanced by each method and which method
wins band-by-band on every metric.

Outputs
-------
  results/per_band_analysis/<dataset>_per_band.csv   per-band numbers
  results/per_band_analysis/<dataset>_snr_gain.png   SNR-gain bar chart
  results/per_band_analysis/<dataset>_ssim.png       per-band SSIM chart
  results/per_band_analysis/<dataset>_psnr.png       per-band PSNR chart
  results/per_band_analysis/<dataset>_summary.csv    enhancement summary

Usage
-----
    python examples/per_band_analysis.py [--dataset hysis|reflectance|both]
                                         [--pct 0.2]
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mnflib import MNF, GeotifImageLoader, MNFConfig, TransformDirection
from mnflib.scores import ssim_score, psnr_score, snr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

IMAGES = {
    "hysis":       os.path.join(_ROOT, "dataset", "Hysis_SW_500x500x125", "HY1SW018901PS011803.tif"),
    "reflectance": os.path.join(_ROOT, "dataset", "reflectance", "reflectance.tif"),
}

RESULTS_DIR = os.path.join(_ROOT, "results", "per_band_analysis")

METHODS = ["next_pixel", "three_pixel", "four_pixel"]
METHOD_LABELS = {
    "next_pixel":   "NPNE",
    "three_pixel":  "TPNE",
    "four_pixel":   "4PNE",
}
METHOD_COLORS = {
    "next_pixel":   "#2196F3",
    "three_pixel":  "#4CAF50",
    "four_pixel":   "#FF5722",
}

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def run_mnf(image: np.ndarray, image_path: str, profile: dict,
            method: str, pct: float) -> np.ndarray:
    lines, bands, samples = image.shape
    cfg = MNFConfig(
        direction=TransformDirection.RUN_BOTH,
        basefilename=image_path,
        bands=bands,
        samples=samples,
        lines=lines,
        percentageOfBandsInInverse=pct,
        noiseMatrixCalculation=method,
        profile=profile,
    )
    transformer = MNF(image, cfg)
    transformer.run()
    return transformer.image


def per_band_metrics(original: np.ndarray, processed: np.ndarray) -> list:
    """Compute SNR / SSIM / PSNR for every spectral band."""
    bands = original.shape[1]
    rows = []
    for b in range(bands):
        orig_band = original[:, b, :]
        proc_band = processed[:, b, :]
        snr_before = snr(orig_band)
        snr_after  = snr(proc_band)
        rows.append({
            "band":       b + 1,
            "snr_before": snr_before,
            "snr_after":  snr_after,
            "snr_gain":   snr_after - snr_before,
            "ssim":       ssim_score(orig_band, proc_band),
            "psnr":       psnr_score(orig_band, proc_band),
        })
    return rows


def count_enhanced(metrics: list) -> int:
    """Bands where SNR improved (gain > 0)."""
    return sum(1 for r in metrics if r["snr_gain"] > 0)


def method_wins(metrics_a: list, metrics_b: list, key: str,
                higher_is_better: bool = True) -> int:
    """Count bands where method A beats method B on *key*."""
    wins = 0
    for a, b in zip(metrics_a, metrics_b):
        if higher_is_better:
            wins += int(a[key] > b[key])
        else:
            wins += int(a[key] < b[key])
    return wins

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _sep(char="─", w=88):
    print(char * w)


def _save_per_band_csv(all_metrics: dict, path: str, bands: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["band"]
    for m in METHODS:
        label = METHOD_LABELS[m]
        fieldnames += [f"{label}_snr_before", f"{label}_snr_after",
                       f"{label}_snr_gain", f"{label}_ssim", f"{label}_psnr"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for b in range(bands):
            row = {"band": b + 1}
            for m in METHODS:
                label = METHOD_LABELS[m]
                r = all_metrics[m][b]
                row[f"{label}_snr_before"] = round(r["snr_before"], 4)
                row[f"{label}_snr_after"]  = round(r["snr_after"],  4)
                row[f"{label}_snr_gain"]   = round(r["snr_gain"],   4)
                row[f"{label}_ssim"]       = round(r["ssim"],       6)
                row[f"{label}_psnr"]       = round(r["psnr"],       4)
            writer.writerow(row)
    print(f"  Saved: {path}")


def _save_summary_csv(summary_rows: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not summary_rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  Saved: {path}")


def _plot_snr_gain(all_metrics: dict, dataset: str, out_dir: str, bands: int):
    band_nums = list(range(1, bands + 1))
    fig, ax = plt.subplots(figsize=(max(12, bands // 4), 5))

    x = np.arange(bands)
    bar_w = 0.28
    offsets = [-bar_w, 0, bar_w]

    for i, method in enumerate(METHODS):
        gains = [all_metrics[method][b]["snr_gain"] for b in range(bands)]
        ax.bar(x + offsets[i], gains, bar_w,
               label=METHOD_LABELS[method], color=METHOD_COLORS[method],
               alpha=0.85, edgecolor="white", linewidth=0.4)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Band")
    ax.set_ylabel("SNR Gain (after − before)")
    ax.set_title(f"{dataset} — Per-Band SNR Gain by Noise Estimation Method")
    ax.set_xticks(x[::max(1, bands // 20)])
    ax.set_xticklabels(band_nums[::max(1, bands // 20)], fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, f"{dataset}_snr_gain.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_metric(all_metrics: dict, metric: str, ylabel: str,
                 dataset: str, out_dir: str, bands: int):
    band_nums = list(range(1, bands + 1))
    fig, ax = plt.subplots(figsize=(max(12, bands // 4), 5))

    for method in METHODS:
        values = [all_metrics[method][b][metric] for b in range(bands)]
        ax.plot(band_nums, values, label=METHOD_LABELS[method],
                color=METHOD_COLORS[method], linewidth=1.4, marker="o",
                markersize=3)

    ax.set_xlabel("Band")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{dataset} — Per-Band {ylabel} by Noise Estimation Method")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, f"{dataset}_{metric}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Per-dataset analysis
# ---------------------------------------------------------------------------

def analyse_dataset(name: str, image_path: str, pct: float) -> dict:
    print(f"\n{'═' * 88}")
    print(f"  Dataset: {name.upper()}")
    print(f"{'═' * 88}")

    loader = GeotifImageLoader(image_path)
    original = loader.get_image()
    lines, bands, samples = original.shape
    print(f"  Shape : {lines} × {bands} bands × {samples}  |  dtype: {original.dtype}")
    print(f"  Inverse bands kept: {int(pct * 100)} %  ({int(pct * bands)} of {bands} bands)\n")

    out_dir = RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    # Run each method and collect per-band metrics
    all_metrics: dict[str, list] = {}
    for method in METHODS:
        label = METHOD_LABELS[method]
        print(f"  Running {label} ...", flush=True)
        processed = run_mnf(original, image_path, loader.profile, method, pct)
        all_metrics[method] = per_band_metrics(original, processed)
        enhanced = count_enhanced(all_metrics[method])
        mean_gain = np.mean([r["snr_gain"] for r in all_metrics[method]])
        mean_ssim = np.mean([r["ssim"]     for r in all_metrics[method]])
        mean_psnr = np.mean([r["psnr"]     for r in all_metrics[method]])
        print(f"    Enhanced bands : {enhanced}/{bands} "
              f"({100 * enhanced / bands:.1f} %)")
        print(f"    Mean SNR gain  : {mean_gain:+.4f}")
        print(f"    Mean SSIM      : {mean_ssim:.6f}")
        print(f"    Mean PSNR      : {mean_psnr:.4f} dB")

    # -----------------------------------------------------------------------
    # Per-band comparison table (console)
    # -----------------------------------------------------------------------
    _sep()
    header = (f"{'Band':>5}  "
              + "  ".join(f"{METHOD_LABELS[m]+' SNR gain':>14}  "
                          f"{METHOD_LABELS[m]+' SSIM':>10}  "
                          f"{METHOD_LABELS[m]+' PSNR':>10}"
                          for m in METHODS))
    print(header)
    _sep()
    for b in range(bands):
        row = f"{b + 1:>5}  "
        for m in METHODS:
            r = all_metrics[m][b]
            row += (f"  {r['snr_gain']:>+14.4f}  "
                    f"{r['ssim']:>10.6f}  "
                    f"{r['psnr']:>10.4f}")
        print(row)

    # -----------------------------------------------------------------------
    # Method-vs-method wins
    # -----------------------------------------------------------------------
    _sep("═")
    print("  Method comparison (band-level wins vs NPNE baseline)\n")
    summary_rows = []
    baseline = "next_pixel"

    for m in METHODS:
        enhanced  = count_enhanced(all_metrics[m])
        snr_wins  = method_wins(all_metrics[m], all_metrics[baseline], "snr_gain") if m != baseline else "—"
        ssim_wins = method_wins(all_metrics[m], all_metrics[baseline], "ssim")     if m != baseline else "—"
        psnr_wins = method_wins(all_metrics[m], all_metrics[baseline], "psnr")     if m != baseline else "—"
        mean_snr_gain = np.mean([r["snr_gain"] for r in all_metrics[m]])
        mean_ssim     = np.mean([r["ssim"]     for r in all_metrics[m]])
        mean_psnr     = np.mean([r["psnr"]     for r in all_metrics[m]])

        label = METHOD_LABELS[m]
        print(f"  {label}")
        print(f"    Bands with SNR improvement : {enhanced}/{bands} ({100*enhanced/bands:.1f} %)")
        if m != baseline:
            print(f"    Bands beating NPNE on SNR gain : {snr_wins}/{bands}")
            print(f"    Bands beating NPNE on SSIM     : {ssim_wins}/{bands}")
            print(f"    Bands beating NPNE on PSNR     : {psnr_wins}/{bands}")
        print(f"    Mean SNR gain : {mean_snr_gain:+.4f}  |  "
              f"Mean SSIM : {mean_ssim:.6f}  |  "
              f"Mean PSNR : {mean_psnr:.4f} dB\n")

        summary_rows.append({
            "dataset":             name,
            "method":              label,
            "bands_total":         bands,
            "bands_snr_improved":  enhanced,
            "pct_snr_improved":    round(100 * enhanced / bands, 1),
            "beats_npne_snr_gain": snr_wins,
            "beats_npne_ssim":     ssim_wins,
            "beats_npne_psnr":     psnr_wins,
            "mean_snr_gain":       round(mean_snr_gain, 4),
            "mean_ssim":           round(mean_ssim,     6),
            "mean_psnr_db":        round(mean_psnr,     4),
        })

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    print("\n  Writing outputs ...")
    _save_per_band_csv(all_metrics, os.path.join(out_dir, f"{name}_per_band.csv"), bands)
    _save_summary_csv(summary_rows,  os.path.join(out_dir, f"{name}_summary.csv"))
    _plot_snr_gain(all_metrics, name, out_dir, bands)
    _plot_metric(all_metrics, "ssim", "SSIM (vs original)", name, out_dir, bands)
    _plot_metric(all_metrics, "psnr", "PSNR dB (vs original)", name, out_dir, bands)

    return {name: summary_rows}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["hysis", "reflectance", "both"],
                        default="both", help="Which dataset to analyse (default: both)")
    parser.add_argument("--pct", type=float, default=0.2,
                        help="Fraction of bands kept in inverse transform (default: 0.2)")
    args = parser.parse_args()

    datasets = (
        list(IMAGES.items()) if args.dataset == "both"
        else [(args.dataset, IMAGES[args.dataset])]
    )

    missing = [(n, p) for n, p in datasets if not os.path.exists(p)]
    if missing:
        for n, p in missing:
            print(f"ERROR: {n} not found at {p}")
        sys.exit(1)

    all_summaries = []
    for name, path in datasets:
        result = analyse_dataset(name, path, args.pct)
        for rows in result.values():
            all_summaries.extend(rows)

    # Combined summary across all datasets
    if len(datasets) > 1:
        _save_summary_csv(all_summaries,
                          os.path.join(RESULTS_DIR, "combined_summary.csv"))

    print(f"\n{'═' * 88}")
    print("  Done. Results in:", RESULTS_DIR)
    print(f"{'═' * 88}\n")


if __name__ == "__main__":
    main()
