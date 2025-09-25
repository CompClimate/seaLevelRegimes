#!/bin/bash 
#SBATCH -p high2

#SBATCH --time=05:30:00
#SBATCH --nodes=1

#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=16

#SBATCH --account=adamgrp

#SBATCH -o dumps/slurm/job_log_%j.output
#SBATCH -e dumps/slurm/job_log_%j.error

# Computing requirements
module purge
module load conda

# Activate the environment
conda activate proc_env

# Ensure the script stops immediately if any Python script fails.
set -e  # Exit immediately if a command exits with a non-zero status
set -x  # Print commands and their arguments

######### Define Variables for the Script #########
# These values are passed as arguments to the script

# Path to your Python script
PYTHON_SCRIPT="/home/djeutsch/Projects/seaLevelRegimes/step_1/embeddings.py" 

# Define the ensemble number, minimum distance, number of neighbors, region, and data scaler
ENSEMBLE=$1 # Ensemble number passed as an argument
MIN_DIST=$2 # Minimum distance passed as an argument
UMAP_KNN=$3 # Number of neighbors passed as an argument
DATA_PATH=$4 # Path to the data file passed as an argument

# Ensure the script exits on error
set -e

# Loop through ech combination of ensemble and minimum distance
echo
for i in {1..5}
do
    # Increment the ensemble number
    ENSEMBLE=$((ENSEMBLE + 1))

    # Create a unique log file for each combination
    
    LOGFILE="/home/djeutsch/Projects/seaLevelRegimes/step_1/dumps/python/ENS${ENSEMBLE}_MD${MIN_DIST}_NN${UMAP_KNN}.log"
    
    # Run the Python script with the specified parameters and redirect output to the log file
    echo "Running embedding Python script for: ensemble = ${ENSEMBLE}, min_dist = ${MIN_DIST}, n_neighbors = ${UMAP_KNN}"
    python -u "$PYTHON_SCRIPT" "$DATA_PATH" "$ENSEMBLE" "$MIN_DIST" "$UMAP_KNN" > "$LOGFILE" &
done
wait # Wait for all background processes to finish

echo
# End of script

