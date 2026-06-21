"""
Plot SNR per band for different MNF configurations.

This script compares SNR across bands for MNF and Line-by-Line MNF
with different noise estimation methods.

Usage:
    python plot_snr_per_band.py
"""

import os
import csv
import shutil
import numpy as np
import matplotlib.pyplot as plt

from mnflib import (
    MNF,
    Line_By_Line_MNF,
    GeotifImageLoader,
    MNFConfig,
    TransformDirection,
    snr as calc_snr,
)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    image_path = os.path.join(_ROOT, "dataset", "reflectance", "reflectance.tif")

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Loading image from {image_path}...")
    image_loader = GeotifImageLoader(image_path)
    original_image = image_loader.get_image()
    lines, bands, samples = original_image.shape

    # User requested sweep from 1 to 12 bands in inverse transform
    inverse_band_counts = [2, 3, 4] #np.arange(3, 10)
    methods     = ["next_pixel", "three_pixel"]
    lbl_methods = ["next_pixel", "three_pixel"]
    
    # Data structure: data[band_index][algo]...
    # MNF: dict by method -> list of SNRs
    # LBL: dict by method -> list of SNRs
    band_data = []
    for _ in range(bands):
        algo_map = {
            "MNF": {m: [] for m in methods},
            "LBL": {m: [] for m in lbl_methods} 
        }
        band_data.append(algo_map)

    print("Starting MNF Sweep...")
    print(f"Inverse Band Counts: {inverse_band_counts}")
    print(f"MNF Methods: {methods}")
    print(f"LBL Methods: {lbl_methods}")

    for n_inv in inverse_band_counts:
        # Calculate percentage ensuring we get exactly n_inv bands
        pct = (n_inv + 1e-5) / bands
        print(f"\nProcessing InverseBands={n_inv} (Pct={pct:.4f})")

        # Config template
        base_config = MNFConfig(
            direction=TransformDirection.RUN_BOTH,
            image_path=image_path,
            bands=bands,
            samples=samples,
            lines=lines,
            percentageOfBandsInInverse=pct,
            noiseMatrixCalculation="next_pixel" 
        )

        # --- ALG 2: Line By Line MNF (Run per METHOD) ---
        for method in lbl_methods:
            print(f"  Running LBL with {method}...")
            cnf = base_config
            cnf.noiseMatrixCalculation = method
            
            current_image_lbl = original_image.copy()
            lbl_mnf = Line_By_Line_MNF(current_image_lbl, cnf)
            result_lbl = lbl_mnf.run()
            
            if lbl_mnf.image is not None:
                for b_idx in range(bands):
                    val = calc_snr(lbl_mnf.image[:, b_idx, :])
                    band_data[b_idx]["LBL"][method].append(val)
            else:
                for b_idx in range(bands):
                    band_data[b_idx]["LBL"][method].append(0)

        # --- ALG 1: Whole Image MNF (Run per METHOD) ---
        for method in methods:
            print(f"  Running MNF with {method}...")
            # Update config for method
            cnf = base_config
            cnf.noiseMatrixCalculation = method
            
            current_image_mnf = original_image.copy()
            mnf = MNF(current_image_mnf, cnf)
            result_mnf = mnf.run()
            
            if mnf.image is not None:
                for b_idx in range(bands):
                    val = calc_snr(mnf.image[:, b_idx, :])
                    band_data[b_idx]["MNF"][method].append(val)
            else:
                for b_idx in range(bands):
                    band_data[b_idx]["MNF"][method].append(0)

    # Plotting
    output_dir = "snr_results"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print(f"\nGenerating plots in {output_dir}/...")

    for b_idx in range(bands):
        plt.figure(figsize=(12, 8))
        
        # Plot MNF (Solid lines)
        for method in methods:
            snr_values = band_data[b_idx]["MNF"][method]
            plt.plot(inverse_band_counts, snr_values, marker='o', linestyle='-', label=f"MNF - {method}")

        # Plot LBL (Dashed lines)
        for method in lbl_methods:
            lbl_values = band_data[b_idx]["LBL"][method]
            plt.plot(inverse_band_counts, lbl_values, marker='x', linestyle='--', linewidth=2, label=f"LBL - {method}")
        
        plt.title(f"Band {b_idx + 1} SNR vs Number of Inverse Bands")
        plt.xlabel("Number of Inverse Bands")
        plt.ylabel("SNR")
        plt.grid(True)
        plt.legend()
        plt.xticks(inverse_band_counts) # Ensure all integer ticks are shown
        
        filename = os.path.join(output_dir, f"band_{b_idx + 1}.png")
        plt.savefig(filename)
        plt.close() # Close figure to free memory

    #     # How many bands score is improving
    # LBL, MNF_ = 0, 0
    # for b_idx in range(bands):
    #     if band_data[b_idx]["MNF"]["three_pixel"] > band_data[b_idx]["MNF"]["next_pixel"]:
    #         MNF_ += 1
    #     if band_data[b_idx]["LBL"]["three_pixel"] > band_data[b_idx]["LBL"]["next_pixel"]:
    #         LBL += 1
    
    # print(f"Improvement in {LBL} out of 40 bands using LBL three_pixel")
    # print(f"Improvement in {MNF_} out of 40 bands using MNF three_pixel")


        # CSV Logging
        csv_filename = os.path.join(output_dir, f"band_{b_idx + 1}.csv")
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Header
            header = ["Inverse Bands", "Percentage"]
            for method in lbl_methods:
                header.append(f"LBL-{method}")
            for method in methods:
                header.append(f"MNF-{method}")
            header.append(f"MNF_improved")
            header.append(f"LBL_improved")
            writer.writerow(header)

            # Rows
            for i, n_inv in enumerate(inverse_band_counts):
                pct = (n_inv + 1e-5) / bands
                
                # Gather MNF values for this row
                mnf_vals = []
                for method in methods:
                    mnf_vals.append(band_data[b_idx]["MNF"][method][i])
                
                # LBL values
                lbl_vals = []
                for method in lbl_methods:
                    lbl_vals.append(band_data[b_idx]["LBL"][method][i])
                
                MNF_improved = [((band_data[b_idx]["MNF"]['three_pixel'][i] - band_data[b_idx]["MNF"]['next_pixel'][i]) / band_data[b_idx]["MNF"]['next_pixel'][i] ) * 100]
                LBL_improved = [((band_data[b_idx]["LBL"]['three_pixel'][i] - band_data[b_idx]["LBL"]['next_pixel'][i]) / band_data[b_idx]["LBL"]['next_pixel'][i] ) * 100]
                
                row = [n_inv, f"{pct:.4f}"] + lbl_vals + mnf_vals + MNF_improved + LBL_improved
                writer.writerow(row)

        
    print("Done.")

if __name__ == "__main__":
    main()
