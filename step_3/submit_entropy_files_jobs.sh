#!/bin/bash
# This script submits jobs to a SLURM scheduler for running the `run_entropy_files.sh` script with various parameters.
# It uses a loop to iterate through different numbers of clusters.

# Define variables for the script
# NUMS_CLUST=(3 5 6 7 10 13 15 16 20 25) # Define the number of clusters (can be adjusted as needed)
NUMS_CLUST=(20) 


echo

for nc in "${NUMS_CLUST[@]}"
do
    # Submit the job to SLURM for each combination
    echo
    echo "Running export entropy job for: num_cluster=${nc}"
    sbatch --job-name=EnF:NC$nc run_entropy_files.sh "$nc"
done

echo
# End of script
