#!/bin/bash
# This script submits jobs to a SLURM scheduler for running the `run_nemi_clusters.sh` script with various parameters.
# It uses a loop to iterate through different numbers of clusters.

# Define variables for the script
# NUMS_CLUSTS=(3 5 6 7 10 13 15 16 20 25) # Define the number of clusters (can be adjusted as needed)
NUMS_CLUSTS=(6 15) #(10 20 25)
BASE_LABEL_IDS=(0 1 2) # Define the base label IDs (can be adjusted as needed)

echo
for nc in "${NUMS_CLUSTS[@]}"
do
    for blid in "${BASE_LABEL_IDS[@]}"
    do
        # Submit the job to SLURM for each combination
        echo "=========================================================================="
        echo "Running BASE LABEL ID, ENTROPY, REGIME Identification Job at ONCE for:"
        echo "#CLUSTERS=${nc}, BASE LABEL ID=${blid}"
        sbatch --job-name=HBLS:$blid:NC$nc run_hbaselabels.sh "$nc" "$blid"
        echo "=========================================================================="
        echo
    done
done

echo
# End of script