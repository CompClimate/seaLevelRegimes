#!/bin/bash
#SBATCH --job-name=LEADING-ORDERED
#SBATCH --account=maikesgrp
#SBATCH --partition=high

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00

#SBATCH -o dumps/lo_balance_log_%j.out
#SBATCH -e dumps/lo_balance_log_%j.err

########################
# Environment
########################
echo
echo "Loading modules and activating conda environment ..."
source /etc/profile.d/modules.sh
module purge
module load conda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /quobyte/maikesgrp/laique/CONDA/conda_envs/nemi_env

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "Environment ready."
echo


########################
# Arguments
########################
echo
echo "Running Ensemble LRP ..."
cd "$SLURM_SUBMIT_DIR"
python -u leading_ordered.py
echo
# End of script