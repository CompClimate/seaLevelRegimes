#!/bin/bash
#SBATCH --job-name=CLUSTERING
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
BASE_DIR="/home/Laique.Djeutchouang/DEVs/BV-Regimes/NEMI/seaLevelRegimes/step_2"

# Path to your Python script
PYTHON_SCRIPT="${BASE_DIR}/hac_clusters.py" 

# Define the ensemble number, minimum distance, number of neighbors, region, and data scaler
DATA_RES=$1 # Data resolution passed as an argument
DATA_FIELD=$2 # Data field passed as an argument
MEMBER=$3 # Ensemble number passed as an argument
UMAP_MD=$4 # Minimum distance passed as an argument
UMAP_NN=$5 # Number of neighbors passed as an argument
HCLUST_N=$6 # Number of neighbors for hierarchical clustering (can be adjusted as needed)
NUM_CLUST=$7 # Number of clusters passed as an argument

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
    LOGFILE="${BASE_DIR}/dumps/python/ENS${MEMBER}_MD${UMAP_MD}_NN${UMAP_NN}_NC${NUM_CLUST}_HC${HCLUST_N}.log"
    
    # Run the Python script with the specified parameters and redirect output to the log file
    echo "Running clustering Python script for: member=${MEMBER}, min_dist=${UMAP_MD}"
	echo "n_neighbors=${UMAP_NN}, num_clusters=${NUM_CLUST}, hclust_n=${HCLUST_N}"
    python -u "$PYTHON_SCRIPT" "$DATA_RES" "$DATA_FIELD" "$MEMBER" "$UMAP_MD" "$UMAP_NN" "$HCLUST_N" "$NUM_CLUST" > "$LOGFILE" &
done
echo
wait # Wait for all background processes to finish

echo
# End of script




