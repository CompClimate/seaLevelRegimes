#!/bin/bash
# This script submits jobs to a SLURM scheduler for running the `run_nemi_clusters.sh` script with various parameters.
# It uses a loop to iterate through different numbers of clusters.

# Define variables for the script
NUM_CLUSTs=(3 5 7 10 13 16 20 25) # Define the number of clusters (can be adjusted as needed)

# Define the data domain and the scaling method
DOMAIN="Global" # It can be adjusted as follows: "Global", NAReg", "SAReg", "CPReg" or "WGReg"
SCALER="Standard" # It can be adjusted as follows: 'Quantile', 'Robust', 'Standard', 'Signed-Log', or 'Power-10'

echo
for nc in "${NUM_CLUSTs[@]}"
do
    # Submit the job to SLURM for each combination
    echo
    echo "Running export NEMI clusters job for:"
    echo " num_cluster=${nc}, region = ${DOMAIN}, data_scaler = ${SCALER}"
    sbatch --job-name=NEMI:NC$nc run_nemi_clusters.sh "$nc" "$DOMAIN" "$SCALER"
done

echo
# End of script

