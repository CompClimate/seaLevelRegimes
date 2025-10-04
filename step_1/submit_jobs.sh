#!/bin/bash

# Runs for all parameters and for 50 ensembles for each param combinations
# This script submits jobs to a SLURM scheduler for running the `run_embeddings.sh` script with various parameters.
# It uses a nested loop to iterate through different combinations of ensemble numbers, minimum distances, and UMAP KNN values.
# The script is designed to be run on a high-performance computing (HPC) cluster.

# Define variables for the script
# Supported by UC Davis SLURM scheduler: mermory expensive jobs when run with large ensembles
ENSEMBLES=(0 5 10 15) # Define the ensemble numbers default to 5 members per ensemble block (can be adjusted as needed). Only if you have enough memory
# ENSEMBLES=(0 5) # For memebers 1 to 10
# ENSEMBLES=(10 15) # For memebers 11 to 20
# ENSEMBLES=(10) # For memebers 11 to 15
# ENSEMBLES=(15) # For memebers 16 to 20
UMAP_MDS=(0.1 0.3 0.5 0.7 0.9) # Define the minimum distances (can be adjusted as needed)
UMAP_NNS=(5 10 50 100 200) # Define the number of neighbors (can be adjusted as needed)

# Seem too large for the job requirements. 
# ENSEMBLES=($(seq 0 5 45)) # Define the ensemble numbers (can be adjusted as needed)

# Define the data domain, path and the scaling method
DATA_PATH="/group/maikesgrp/laique/NOAA/nemis/CM4X-p125/inputs/glob_scaled_CM4X-p125_BV_budget_time_mean.parquet" # Path to your data file

echo

for umap_md in "${UMAP_MDS[@]}"
do
	for umap_nn in "${UMAP_NNS[@]}"
	do
		for member in "${ENSEMBLES[@]}"
		do
			# Submit the job to SLURM for each combination
			echo
			echo "Running embedding job for: block_5_ensemble_num = ${member}, min_dist = ${umap_md}, n_neighbors = ${umap_nn}"
			sbatch --job-name=Emb-$member.$umap_md.$umap_nn run_embeddings.sh "$member" "$umap_md" "$umap_nn" "$DATA_PATH" 
		done	
	done
done

echo
# End of script