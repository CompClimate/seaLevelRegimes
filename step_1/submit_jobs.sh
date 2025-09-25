#!/bin/bash

# Runs for all parameters and for 50 ensembles for each param combinations
# This script submits jobs to a SLURM scheduler for running the `run_embeddings.sh` script with various parameters.
# It uses a nested loop to iterate through different combinations of ensemble numbers, minimum distances, and UMAP KNN values.
# The script is designed to be run on a high-performance computing (HPC) cluster.

# Define variables for the script
# Supported by UC Davis SLURM scheduler 
ENSEMBLES=(0 5) # Define the ensemble numbers default to 5 (can be adjusted as needed)
# ENSEMBLES=(15) # For one member only
MIN_DISTANCES=(0.1 0.3 0.5 0.7 0.9) # Define the minimum distances (can be adjusted as needed)
UMAP_KNNS=(5 10 50 100 200) # Define the number of neighbors (can be adjusted as needed)

# Seem too large for the job requirements. 
# ENSEMBLES=($(seq 0 5 45)) # Define the ensemble numbers (can be adjusted as needed)

# Define the data domain, path and the scaling method
DATA_PATH="/group/maikesgrp/laique/NOAA/nemis/CM4X-p125/inputs/glob_scaled_CM4X-p125_BV_budget_time_mean.parquet" # Path to your data file

echo

for md in "${MIN_DISTANCES[@]}"
do
	for u_knn in "${UMAP_KNNS[@]}"
	do
		for ens in "${ENSEMBLES[@]}"
		do
			# Submit the job to SLURM for each combination
			echo
			echo "Running embedding job for: block_5_ensemble_num = ${ens}, min_dist = ${md}, n_neighbors = ${u_knn}"
			sbatch --job-name=Emb-$ens.$md.$u_knn run_embeddings.sh "$ens" "$md" "$u_knn" "$DATA_PATH"
		done	
	done
done

echo
# End of script