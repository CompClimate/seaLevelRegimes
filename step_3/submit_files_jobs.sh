#!/bin/bash
# This script submits jobs to a SLURM scheduler for running the `run_entropy_files.sh` script with various parameters.
# It uses a loop to iterate through different numbers of clusters.

# Define variables for the script
NUMS_CLUST=(3 5 6 7 10 13 15 16 20 25) # Define the number of clusters (can be adjusted as needed)

# Give the resolution and field for the data
DATA_RES="p25"
DATA_FIELD="dynamics" # 'statics' (mean) or 'dynamics' (e.g.; monthly climatology) 


echo

for nc in "${NUMS_CLUST[@]}"
do
    # Submit the job to SLURM for each combination
    echo
    echo "Running export entropy job for: num_cluster=${nc}"
    sbatch --job-name=EnF:NC$nc run_entropy_files.sh "$DATA_RES" "$DATA_FIELD" "$nc"
done

echo
# End of script
