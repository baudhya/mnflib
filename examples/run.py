"""
Batch runner script for executing MNF transforms with multiple configurations.

This script uses the mnf CLI command to run multiple MNF configurations
and logs the output to a file.

Usage:
    python run.py
"""

import subprocess

opt_1 = [f"--inverse-bands-percentage {i/40 + 0.00001}" for i in range(2, 5)] 
# Use the mnf CLI command instead of calling main.py directly
init_cmd = ["mnf", "--image-path=/home/gollum/Desktop/mnf/dataset/reflectance/reflectance.tif"]
# opt_2 = ["--noise-matrix next_pixel", "--noise-matrix three_pixel"]
opt_2 = ["--noise-matrix next_pixel", "--noise-matrix three_pixel"]
output_file = "all_output_results.txt"


with open(output_file, "w") as f:
    f.write("")


# with open(output_file, "a") as f:
#     for optfirst in opt_1:
#         for optsecond in opt_2:
#             cmd = init_cmd + optfirst.split() + optsecond.split()
#             f.write("Running: "+ " ".join(cmd) + "\n")
#             result = subprocess.run(
#                 cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.STDOUT,
#                 text=True
#             )
#             f.write(result.stdout)
#             f.write("\n----------------------------------------------------------------------------------------\n\n")


with open(output_file, "a") as f:
    for optfirst in opt_1:
        for optsecond in opt_2:
            cmd = init_cmd + optfirst.split() + optsecond.split() + ["--line-by-line"]
            f.write("Running: "+ " ".join(cmd) + "\n")
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            f.write(result.stdout)
            f.write("\n----------------------------------------------------------------------------------------\n\n")

print("Script Completed.")