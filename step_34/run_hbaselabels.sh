#!/bin/bash
#SBATCH --job-name=HBASELABELS
#SBATCH --account=publicgrp
#SBATCH --partition=low

#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=300G
#SBATCH --time=24:00:00

#SBATCH -o dumps/slurm/hbaselabels_log_%j.out
#SBATCH -e dumps/slurm/hbaselabels_log_%j.err

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

PYTHON_SCRIPT_NUM1="${BASE_DIR}/hbaselabels.py"   # Step 1

NUM_CLUST=$1 # Number of clusters passed as an argument
BLID=$2 # Base label ID passed as an argument

echo
# Ensure the script exits on error
set -e

######### Run Scripts Sequentially #########

echo "Running BASE LABEL ID script for:"
echo "  num_cluster=${NUM_CLUST}"
python -u "$PYTHON_SCRIPT_NUM1" "$NUM_CLUST" "$BLID"
echo "Step 1 complete."
echo

# End of script