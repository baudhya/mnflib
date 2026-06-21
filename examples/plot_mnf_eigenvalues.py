"""
Plot MNF eigenvalues for different noise estimation methods.

This script compares eigenvalue spectra across MNF and Line-by-Line MNF
with different noise estimation methods.

Usage:
    python plot_mnf_eigenvalues.py
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from mnflib import (
    MNF,
    Line_By_Line_MNF,
    GeotifImageLoader,
    MNFConfig,
    TransformDirection,
    noise_added_image,
)

def main():
    # Hardcoded path as per user context / implied requirement
    # Ideally this would be an argument, but for this specific "make a script" request I'll use the known path
    image_path = "C:\\Users\\ISDSS01\\Desktop\\Siddharth\\reflectance\\reflectance.tif"
    # image_path = "C:\\Users\\ISDSS01\\Desktop\\Siddharth\\Hysis_SW_500x500x125\\HY1SW018901PS011803.tif"
    image_path = "/home/gollum/Desktop/mnf/dataset/reflectance/reflectance.tif"

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Loading image from {image_path}...")
    image_loader = GeotifImageLoader(image_path)
    original_image = image_loader.get_image()
    
    # Using original image (or noisy if needed, but usually MNF is run on raw or noisy data)
    # The user request implies comparing methods on valid data.
    # main.py adds noise for testing; I'll stick to original or raw if possible.
    # main.py does: noisy_image = original_image if add_noise=0.
    # I'll use raw image logic similar to main.py defaults (no extra noise added here unless requested, 
    # user didn't request synthetic noise, just comparison on "MNF").
    
    lines, bands, samples = original_image.shape
    
    methods = ["next_pixel", "three_pixel"]
    results = {}

    for method in methods:
        print(f"\nRunning MNF with noise method: {method}")
        
        # Configure MNF
        # Direction: FORWARD is sufficient to compute eigenvalues
        mnf_config = MNFConfig(
            direction=TransformDirection.RUN_FORWARD, 
            image_path=image_path,
            bands=bands,
            samples=samples,
            lines=lines,
            percentageOfBandsInInverse=0.2, # Default, doesn't matter for forward
            noiseMatrixCalculation=method
        )
        
        # Instantiate and Run
        # Note: We pass original_image.copy() because MNF modifies in-place
        mnf = MNF(original_image.copy(), mnf_config)
        result = mnf.run()
        
        if result.eigenvalues is not None:
            results[method] = result.eigenvalues
        else:
            print(f"Warning: No eigenvalues returned for {method}")

    for method in ["next_pixel", "three_pixel"]:
        print(f"\nRunning MNF with noise method: {method}")
        
        # Configure MNF
        # Direction: FORWARD is sufficient to compute eigenvalues
        mnf_config = MNFConfig(
            direction=TransformDirection.RUN_FORWARD, 
            image_path=image_path,
            bands=bands,
            samples=samples,
            lines=lines,
            percentageOfBandsInInverse=0.2, # Default, doesn't matter for forward
            noiseMatrixCalculation=method
        )
        
        # Instantiate and Run
        # Note: We pass original_image.copy() because MNF modifies in-place
        mnf = Line_By_Line_MNF(original_image.copy(), mnf_config)
        result = mnf.run()
        
        if result.eigenvalues is not None:
            results["LBL_" + method] = result.eigenvalues
        else:
            print(f"Warning: No eigenvalues returned for {method}")

    # Plotting
    if not results:
        print("No results to plot.")
        return

    plt.figure(figsize=(10, 6))
    
    for method, eigvals in results.items():
        # Plotting log eigenvalues is often more informative for MNF
        plt.plot(eigvals, label=f"{method}", linestyle='-')
        # Alternatively, plot raw if preferred, but usually MNF eigenvalues span orders of magnitude
        # User said "simple graph". I'll stick to linear plot as per standard unless they ask for log.
        # However, to be safe and match standard HSI viz, I'll log-scale the Y axis if they are very widespread.
        # But for "simple graph", I will plot raw values first.
        # Wait, usually MNF eigenvalues are plotted on semi-log or just log Y. 
        # I'll stick to raw plot with y-log scale if needed, or just standard plot function.
        # Let's use standard plot but maybe add a semilogy version?
        # I'll enable grid.
    
    plt.xlabel("Component Number")
    plt.ylabel("Eigenvalue")
    plt.title("MNF Eigenvalues Comparison by Noise Method")
    # plt.yscale('log') # Log scale is almost always better for eigenspectra
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    output_file = "mnf_eigenvalue_comparison.png"
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")
    # plt.show() # Cannot show in headless, stick to save

if __name__ == "__main__":
    main()
