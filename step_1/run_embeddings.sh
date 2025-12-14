#!/bin/bash
#SBATCH --job-name=EMBEDDING
#SBATCH --account=gfdl_o
#SBATCH --partition=analysis 
#SBATCH --constraint=bigmem 

#SBATCH --nodes=1
#SBATCH --mem=700G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2

#SBATCH -o dumps/slurm/nemi_job_%j.out
#SBATCH -e dumps/slurm/nemi_job_%j.err


# Computing requirements
module purge
module load conda

# Activate the desired environment
conda activate /work/lnd/ODRI/CONDA/conda_envs/nemi_env

######### Define Variables for the Script #########
# These values are passed as arguments to the script

# Base directory for the project
BASE_DIR="/home/Laique.Djeutchouang/DEVs/BV-Regimes/NEMI/seaLevelRegimes/step_1"

# Path to your Python script
PYTHON_SCRIPT="${BASE_DIR}/embeddings.py"

# Define the ensemble number, minimum distance, number of neighbors, region, and data scaler
MEMBER=$1 # Ensemble number passed as an argument
UMAP_MD=$2 # Minimum distance passed as an argument
UMAP_NN=$3 # Number of neighbors passed as an argument
DATA_RES=$4 # Data resolution passed as an argument
DATA_FIELD=$5 # Data field passed as an argument

set -e  # Exit immediately if a command exits with a non-zero status

# Loop through ech combination of ensemble and minimum distance
echo
for i in {1..5}
do
    # Increment the ensemble number
    MEMBER=$((MEMBER + 1))

    # Create a unique log file for each combination

    LOGFILE="${BASE_DIR}/dumps/python/ENS${MEMBER}_MD${UMAP_MD}_NN${UMAP_NN}.log"

    # Run the Python script with the specified parameters and redirect output to the log file
    echo "Running embedding Python script for: member=${MEMBER}, min_dist=${UMAP_MD}, n_neighbors=${UMAP_NN}"
    python -u "$PYTHON_SCRIPT" "$DATA_RES" "$DATA_FIELD" "$MEMBER" "$UMAP_MD" "$UMAP_NN" > "$LOGFILE" &
done
echo
wait # Wait for all background processes to finish

echo
# End of script

