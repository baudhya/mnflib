#!/usr/bin/env python3
"""
Generate noise-variance comparison tables (Table 1 & Table 2 from the paper).

Runs NPNE (next_pixel) and TPNE (three_pixel) noise estimation under both
Global MNF and Line-by-Line (LBL) architectures across a configurable set of
inverse-band counts and spectral bands, then prints the results as formatted
tables and optionally saves them as CSV files.

Usage
-----
    python examples/generate_results_table.py --image-path dataset/reflectance/reflectance.tif
    python examples/generate_results_table.py \\
        --image-path dataset/reflectance/reflectance.tif \\
        --bands 30 33 \\
        --inv-bands 2 3 4 \\
        --save-csv
"""

import argparse
import os
import sys
import numpy as np

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnflib.dataloader import GeotifImageLoader
from mnflib.data_structures import MNFConfig, TransformDirection
from mnflib.mnf import MNF
from mnflib.mnf_linebyline import Line_By_Line_MNF


# ── core helpers ─────────────────────────────────────────────────────────────

def _make_config(base_path, profile, lines, total_bands, samples,
                 noise_method, n_inv_bands):
    return MNFConfig(
        direction=TransformDirection.RUN_BOTH,
        basefilename=base_path,
        bands=total_bands,
        samples=samples,
        lines=lines,
        percentageOfBandsInInverse=n_inv_bands / total_bands,
        noiseMatrixCalculation=noise_method,
        profile=profile,
    )


def run_global_mnf(image, cfg):
    mnf = MNF(image.copy(), cfg)
    mnf.run()
    return mnf.image


def run_lbl_mnf(image, cfg):
    lbl = Line_By_Line_MNF(image.copy(), cfg)
    lbl.run()
    return lbl.image


def band_variance(image, band_1indexed):
    """Variance of one spectral band from a (lines, bands, samples) array."""
    return float(np.var(image[:, band_1indexed - 1, :]))


# ── experiment runner ─────────────────────────────────────────────────────────

def run_all(image_path, bands_of_interest, inv_bands_list):
    """
    Run all (architecture × noise_method × inv_bands) combinations.

    Returns
    -------
    dict
        Keys: (architecture, band, n_inv, noise_method)
        Values: reconstructed-band variance (float)
    """
    loader = GeotifImageLoader(image_path)
    image = loader.get_image()
    profile = loader.profile
    lines, total_bands, samples = image.shape

    print(f"Image shape  : lines={lines}, bands={total_bands}, samples={samples}")
    print(f"Bands tested : {bands_of_interest}")
    print(f"Inv. bands   : {inv_bands_list}")
    print()

    architectures = [
        ("global", "Global MNF", run_global_mnf),
        ("lbl",    "LBL",        run_lbl_mnf),
    ]
    noise_methods = [
        ("next_pixel",   "NPNE"),
        ("three_pixel",  "TPNE"),
    ]

    results = {}
    total_runs = len(architectures) * len(noise_methods) * len(inv_bands_list)
    run_num = 0

    for arch_key, arch_label, arch_fn in architectures:
        for noise_method, noise_label in noise_methods:
            for n_inv in inv_bands_list:
                run_num += 1
                tag = f"[{run_num}/{total_runs}]"
                print(f"{tag} {arch_label} | {noise_label} | inv_bands={n_inv}")

                cfg = _make_config(
                    image_path, profile, lines, total_bands, samples,
                    noise_method, n_inv,
                )
                recon = arch_fn(image, cfg)

                for band in bands_of_interest:
                    var = band_variance(recon, band)
                    results[(arch_key, band, n_inv, noise_method)] = var
                    print(f"        Band {band:>2} variance = {var:.4f}")
                print()

    return results


# ── display helpers ───────────────────────────────────────────────────────────

def _build_table_rows(results, arch_key, bands, inv_bands_list):
    """Return list of (band, n_inv, npne_var, tpne_var) for one architecture."""
    rows = []
    for band in bands:
        for n_inv in inv_bands_list:
            npne = results.get((arch_key, band, n_inv, "next_pixel"), float("nan"))
            tpne = results.get((arch_key, band, n_inv, "three_pixel"), float("nan"))
            rows.append((band, n_inv, npne, tpne))
    return rows


def print_table(title, rows):
    sep = "-" * 52
    print(f"\n{title}")
    print("=" * 52)
    print(f"  {'Band':>4}  {'Inv. Bands':>10}  {'NPNE':>10}  {'TPNE':>10}")
    print(sep)
    prev_band = None
    for band, n_inv, npne, tpne in rows:
        band_str = str(band) if band != prev_band else ""
        print(f"  {band_str:>4}  {n_inv:>10}  {npne:>10.4f}  {tpne:>10.4f}")
        prev_band = band
    print("=" * 52)


def save_csv(filepath, title, rows):
    with open(filepath, "w") as f:
        f.write(f"# {title}\n")
        f.write("Band,Inv_Bands,NPNE,TPNE\n")
        for band, n_inv, npne, tpne in rows:
            f.write(f"{band},{n_inv},{npne:.4f},{tpne:.4f}\n")
    print(f"  Saved → {filepath}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Table 1 (LBL) and Table 2 (Global MNF) from the paper: "
            "noise-variance comparison between NPNE and TPNE across inverse-band counts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image-path", required=True,
        help="Path to the hyperspectral GeoTIFF image.",
    )
    parser.add_argument(
        "--bands", nargs="+", type=int, default=[30, 33], metavar="BAND",
        help="Spectral bands to include (1-indexed).",
    )
    parser.add_argument(
        "--inv-bands", nargs="+", type=int, default=[2, 3, 4], metavar="N",
        help="Number of inverse bands (eigenvectors) to test.",
    )
    parser.add_argument(
        "--save-csv", action="store_true",
        help="Save Table 1 and Table 2 as CSV files next to the image.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        sys.exit(f"Error: image not found: {args.image_path}")

    results = run_all(args.image_path, args.bands, args.inv_bands)

    lbl_rows    = _build_table_rows(results, "lbl",    args.bands, args.inv_bands)
    global_rows = _build_table_rows(results, "global", args.bands, args.inv_bands)

    print_table("Table 1: Noise estimation variance using the LBL",    lbl_rows)
    print_table("Table 2: Noise estimation variance using the Global MNF", global_rows)

    if args.save_csv:
        out_dir = os.path.dirname(os.path.abspath(args.image_path))
        save_csv(
            os.path.join(out_dir, "results_table1_lbl.csv"),
            "Noise estimation variance using the LBL",
            lbl_rows,
        )
        save_csv(
            os.path.join(out_dir, "results_table2_global_mnf.csv"),
            "Noise estimation variance using the Global MNF",
            global_rows,
        )


if __name__ == "__main__":
    main()
