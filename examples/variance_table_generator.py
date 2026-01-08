import re
from collections import defaultdict

def parse_results_file(filename):
    """Parse the all_output_results.txt file and extract variance data."""
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Split by separator lines to get individual runs
    runs = content.split('----------------------------------------------------------------------------------------')
    
    # Dictionary to store variance data
    # Structure: {config: {band_number: variance_value}}
    variance_data = defaultdict(dict)
    
    for run in runs:
        if not run.strip():
            continue
        
        # Extract configuration details
        noise_match = re.search(r'--noise-matrix\s+(\w+)', run)
        inverse_match = re.search(r'--inverse-bands-percentage\s+([\d.]+)', run)
        line_by_line_match = re.search(r'--line-by-line', run)
        
        if not noise_match or not inverse_match:
            continue
            
        noise_method = noise_match.group(1)
        inverse_pct = float(inverse_match.group(1))
        
        # Determine algorithm type
        if line_by_line_match:
            algorithm = "LBL"
        else:
            algorithm = "MNF"
        
        # Create config label with algorithm type
        config_label = f"{algorithm}_{noise_method}_{inverse_pct:.4f}"
        
        # Extract variance values for each band
        variance_pattern = r'Variance for Band (\d+) : ([\d.e-]+)'
        variance_matches = re.findall(variance_pattern, run)
        
        for band_num, variance_val in variance_matches:
            band_num = int(band_num)
            variance_val = float(variance_val)
            variance_data[config_label][band_num] = variance_val
    
    return variance_data

def create_variance_table(variance_data):
    """Create a formatted table from variance data."""
    
    # Get all band numbers (should be 1-40)
    all_bands = sorted(set(band for config in variance_data.values() for band in config.keys()))
    
    # Get all configurations
    configs = sorted(variance_data.keys())
    
    # Print table header
    header = "| Band | " + " | ".join(configs) + " |"
    separator = "|" + "|".join(["-" * 6] + ["-" * 30 for _ in configs]) + "|"
    
    print(header)
    print(separator)
    
    # Print data rows
    for band in all_bands:
        row = f"| {band:4d} |"
        for config in configs:
            variance = variance_data[config].get(band, 0.0)
            row += f" {variance:28.6e} |"
        print(row)

def create_markdown_table(variance_data, output_file):
    """Create a markdown table and save to file."""
    
    # Get all band numbers
    all_bands = sorted(set(band for config in variance_data.values() for band in config.keys()))
    
    # Get all configurations
    configs = sorted(variance_data.keys())
    
    lines = []
    
    # Create header
    header = "| Band | " + " | ".join(configs) + " |"
    separator = "|" + "|".join([":----:"] + [":----------------------------:" for _ in configs]) + "|"
    
    lines.append(header)
    lines.append(separator)
    
    # Create data rows
    for band in all_bands:
        row = f"| {band:4d} |"
        for config in configs:
            variance = variance_data[config].get(band, 0.0)
            row += f" {variance:.6e} |"
        lines.append(row)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    return lines

def create_csv_table(variance_data, output_file):
    """Create a CSV table in long format with separate columns for parameters."""
    
    # Get all band numbers
    all_bands = sorted(set(band for config in variance_data.values() for band in config.keys()))
    
    lines = []
    
    # Create header
    header = "Band,Algorithm,Noise_Type,Inverse_Bands_Percentage,Variance"
    lines.append(header)
    
    # Create data rows - one row per band per configuration
    for config in sorted(variance_data.keys()):
        # Parse configuration: format is "algorithm_noise_type_percentage"
        # e.g., "MNF_next_pixel_0.0500" or "LBL_three_pixel_0.0750"
        # We need to extract: algorithm (MNF or LBL), noise_type (next_pixel or three_pixel), percentage
        
        # Split by underscore and work backwards
        parts = config.split('_')
        
        # Last part is always the percentage
        inverse_pct = parts[-1]
        
        # First part is always the algorithm (MNF or LBL)
        algorithm = parts[0]
        
        # Middle parts form the noise type (next_pixel or three_pixel)
        noise_type = '_'.join(parts[1:-1])
        
        for band in all_bands:
            variance = variance_data[config].get(band, 0.0)
            row = f"{band},{algorithm},{noise_type},{inverse_pct},{variance}"
            lines.append(row)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    return lines

if __name__ == "__main__":
    # Parse the results file
    input_file = "all_output_results.txt"
    variance_data = parse_results_file(input_file)
    
    print("="*80)
    print("VARIANCE TABLE FOR ALL BANDS")
    print("="*80)
    print()
    
    # Display table in console
    create_variance_table(variance_data)
    
    print()
    print("="*80)
    
    # Create markdown table
    md_output = "variance_table.md"
    create_markdown_table(variance_data, md_output)
    print(f"\nMarkdown table saved to: {md_output}")
    
    # Create CSV table
    csv_output = "variance_table.csv"
    create_csv_table(variance_data, csv_output)
    print(f"CSV table saved to: {csv_output}")
