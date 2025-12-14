#!/bin/bash

# Runs for all parameters and for 50 ensembles for each param combinations
# This script submits jobs to a SLURM scheduler for running the `run_cluster.sh` script with various parameters.
# It uses a nested loop to iterate through different combinations of ensemble size, UMAP minimum distances, UMAP NN values, and number of clusters.
# The script is designed to be run on a high-performance computing (HPC) cluster.

# Define variables for the script
# Supported by UC Davis SLURM scheduler 
ENSEMBLES=(0 5 10 15) # Define the ensemble numbers default to 5 members per ensemble block (can be adjusted as needed). Only if you have enough memory
# ENSEMBLES=(0 5) # For memebers 1 to 10
# ENSEMBLES=(15) # For memebers 16 to 20
UMAP_MDS=(0.1 0.3 0.5 0.7 0.9) # Define the minimum distances (can be adjusted as needed)
UMAP_NNS=(5 10 50 100 200) # Define the number of neighbors (can be adjusted as needed)
NUM_CLUSTS=(3 5 6 7 10 13 15 16 20 25) # Define the number of clusters (can be adjusted as needed)
# NUM_CLUSTS=(3 10) # To run non completed clusterings
HCLUST_N=40 # Define the number of neighbors for hierarchical clustering (can be adjusted as needed)

# Give the resolution and field for the data
DATA_RES="p25"
DATA_FIELD="dynamics" # 'statics' (mean) or 'dynamics' (e.g.; monthly climatology) 

# Seem too large for the job requirements. 
# ENSEMBLES=($(seq 0 5 45)) # Define the ensemble numbers (can be adjusted as needed)

echo

for umap_md in "${UMAP_MDS[@]}"
do
	for umap_nn in "${UMAP_NNS[@]}"
	do
		for member in "${ENSEMBLES[@]}"
		do
			for nc in "${NUM_CLUSTS[@]}"
			do
				# Submit the job to SLURM for each combination
				echo
				echo "Running clustering job for: block_5_ensemble_num=${member}, min_dist=${umap_md}"
				echo "n_neighbors ${umap_nn}, num_clusters=${nc}, hclust_n=${HCLUST_N}"
				sbatch --job-name=HAC-$member.$umap_md.$umap_nn.$nc run_hac_clusters.sh "$DATA_RES" "$DATA_FIELD" "$member" "$umap_md" "$umap_nn" "$HCLUST_N" "$nc"
				echo
			done
		done	
	done
done

echo
# End of script


