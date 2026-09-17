#!/bin/bash 
#SBATCH -p med2

#SBATCH --time=10:00:00
#SBATCH --nodes=1

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G

#SBATCH --account=adamgrp

#SBATCH -o dumps/slurm/job_log_%j.output
#SBATCH -e dumps/slurm/job_log_%j.error

# Computing requirements
module purge
module load conda

# Activate the environment
conda activate proc_env

######### Define Variables for the Script #########
# These values are passed as arguments to the script

# Path to your Python script
PYTHON_SCRIPT="/home/djeutsch/Projects/seaLevelRegimes/step_2/hac_clusters.py" 

# Define the ensemble number, minimum distance, number of neighbors, region, and data scaler
MEMBER=$1 # Ensemble number passed as an argument
UMAP_MD=$2 # Minimum distance passed as an argument
UMAP_NN=$3 # Number of neighbors passed as an argument
NUM_CLUST=$4 # Number of clusters passed as an argument
HCLUST_N=$5 # Number of neighbors for hierarchical clustering (can be adjusted as needed)

# Ensure the script exits on error
set -e

# Loop through each combination of ensemble and minimum distance
echo
for i in {1..5}
do
    echo
    # Increment the ensemble number
    MEMBER=$((MEMBER + 1))

    # Create a unique log file for each combination
    LOGFILE="/home/djeutsch/Projects/seaLevelRegimes/step_2/dumps/python/ENS${MEMBER}_MD${UMAP_MD}_NN${UMAP_NN}_NC${NUM_CLUST}_HC${HCLUST_N}.log"
    
    # Run the Python script with the specified parameters and redirect output to the log file
    echo "Running clustering Python script for: member = ${MEMBER}, min_dist = ${UMAP_MD}"
	echo "n_neighbors = ${UMAP_NN}, num_clusters=${NUM_CLUST}, hclust_n = ${HCLUST_N}"
    python -u "$PYTHON_SCRIPT" "$MEMBER" "$UMAP_MD" "$UMAP_NN" "$NUM_CLUST" "$HCLUST_N" > "$LOGFILE" &
done
wait # Wait for all background processes to finish

echo
# End of script




