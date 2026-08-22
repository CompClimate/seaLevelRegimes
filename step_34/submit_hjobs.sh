#!/bin/bash
# This script submits jobs to a SLURM scheduler for running the `run_nemi_clusters.sh` script with various parameters.
# It uses a loop to iterate through different numbers of clusters.

# Define variables for the script
# NUMS_CLUSTS=(3 5 6 7 10 13 15 16 20 25) # Define the number of clusters (can be adjusted as needed)
NUMS_CLUSTS=(6 15) #(10 20 25)

echo
for nc in "${NUMS_CLUSTS[@]}"
do
    # Submit the job to SLURM for each combination
    echo "========================================================================="
    echo "Running BASE LABEL ID, ENTROPY, REGIME Identification Job at ONCE for:"
    echo "#Clusters=${nc}, BASE LABEL ID=${blid}"
    sbatch --job-name=HJOBS:$blid:NC$nc run_hjobs.sh "$nc"
    echo "========================================================================="
    echo 
done

echo
# End of script