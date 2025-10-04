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
PYTHON_SCRIPT="/home/djeutsch/Projects/seaLevelRegimes/step_1/embeddings.py" 

# Define the ensemble number, minimum distance, number of neighbors, region, and data scaler
MEMBER=$1 # Ensemble number passed as an argument
UMAP_MD=$2 # Minimum distance passed as an argument
UMAP_NN=$3 # Number of neighbors passed as an argument
DATA_PATH=$4 # Path to the data file passed as an argument

set -e  # Exit immediately if a command exits with a non-zero status

# Loop through ech combination of ensemble and minimum distance
echo
for i in {1..5}
do
    # Increment the ensemble number
    MEMBER=$((MEMBER + 1))

    # Create a unique log file for each combination
    
    LOGFILE="/home/djeutsch/Projects/seaLevelRegimes/step_1/dumps/python/ENS${MEMBER}_MD${UMAP_MD}_NN${UMAP_NN}.log"
    
    # Run the Python script with the specified parameters and redirect output to the log file
    echo "Running embedding Python script for: member = ${MEMBER}, min_dist = ${MIN_DIST}, n_neighbors = ${UMAP_NN}"
    python -u "$PYTHON_SCRIPT" "$DATA_PATH" "$MEMBER" "$UMAP_MD" "$UMAP_NN" > "$LOGFILE" &
done
wait # Wait for all background processes to finish

echo
# End of script

