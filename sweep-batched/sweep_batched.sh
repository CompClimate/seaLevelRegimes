#!/bin/bash
#SBATCH --job-name=CPU-DBSCAN
#SBATCH --account=gfdl_o
#SBATCH --partition=analysis 
#SBATCH --constraint=bigmem 
##SBATCH --partition=gpu

##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1010G
#SBATCH --time=48:00:00

#SBATCH -o dumps/ppan/clim_slog_%j.out
#SBATCH -e dumps/ppan/clim_slog_%j.err

# set -euo pipefail  # Moved to top: exit on error, unbound vars, and pipe failures

########################
# Environment
########################
echo
echo "Loading modules and activating conda environment ..."
module purge
module load conda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /work/lnd/ODRI/CONDA/conda_envs/nemi_env

echo "Environment ready."
echo

if [ "$#" -ne 4 ]; then
    echo "Usage: sbatch sweep_batched.sh <input_file> <output_dir> <md> <nn>" >&2
    exit 1
fi

INPUT_FILE=$1
OUTPUT_DIR=$2
MD=$3
NN=$4

OUTPUT_DIR="${OUTPUT_DIR}/md${MD}_nn${NN}_JOB${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_DIR}"

###############################################################################
# Run sweep_batched.py
###############################################################################
python -u sweep_batched.py "${INPUT_FILE}" "${OUTPUT_DIR}"


# End of script

