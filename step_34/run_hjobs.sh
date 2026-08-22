#!/bin/bash
#SBATCH --job-name=HJOBS
#SBATCH --account=publicgrp
#SBATCH --partition=low

#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=300G
#SBATCH --time=24:00:00

#SBATCH -o dumps/slurm/hentropy_id_log_%j.out
#SBATCH -e dumps/slurm/hentropy_id_log_%j.err

# set -euo pipefail  # Moved to top: exit on error, unbound vars, and pipe failures

########################
# Environment
########################
echo
echo "Loading modules and activating conda environment ..."
module purge
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /quobyte/maikesgrp/laique/CONDA/conda_envs/nemi_env
echo "Environment ready."
echo

######### Define Variables #########
BASE_DIR="/home/djeutsch/Projects/seaLevelRegimes/step_34"

PYTHON_SCRIPT_NUM2="${BASE_DIR}/hentropy.py"       # Step 2
PYTHON_SCRIPT_NUM3="${BASE_DIR}/hidentifier.py"    # Step 3

NUM_CLUST=$1 # Number of clusters passed as an argument

echo
# Ensure the script exits on error
set -e

######### Run Scripts Sequentially #########

echo "Running ENTROPY script for:"
echo "  num_cluster=${NUM_CLUST}"
python -u "$PYTHON_SCRIPT_NUM2" "$NUM_CLUST"
echo "Step 2 complete."
echo

echo "Running REGIME IDENTIFICATION script for:"
echo "  num_cluster=${NUM_CLUST}"
python -u "$PYTHON_SCRIPT_NUM3" "$NUM_CLUST"
echo "Step 3 complete."
echo

# End of script