#!/bin/bash
#SBATCH --job-name=CLUSTERING
#SBATCH --account=gfdl_o
#SBATCH --partition=analysis 
#SBATCH --constraint=bigmem 

#SBATCH --nodes=1
#SBATCH --mem=512G
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
BASE_DIR="/home/Laique.Djeutchouang/DEVs/BV-Regimes/NEMI/seaLevelRegimes/step_3"

# Path to your Python script
PYTHON_SCRIPT="${BASE_DIR}/entropy_baselabels.py" 

DATA_RES=$1 # Data resolution passed as an argument
DATA_FIELD=$2 # Data field passed as an argument
BLID=$3 # Base label ID passed as an argument
NUM_CLUST=$4 # Number of clusters passed as an argument

echo
# Ensure the script exits on error
set -e

# Create a unique log file for each combination
LOGFILE="${BASE_DIR}/dumps/python/ent_nclust_NC${NUM_CLUST}_BLID${BLID}.log"
echo "Running entropy base labels Python script for:"
echo " num_cluster=${NUM_CLUST}, base_label_id=${BLID}"
python -u "$PYTHON_SCRIPT" "$DATA_RES" "$DATA_FIELD" "$BLID" "$NUM_CLUST" > "$LOGFILE"

echo
# End of script



