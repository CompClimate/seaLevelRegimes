#!/bin/bash
#SBATCH --job-name=UMAP_EMBED
#SBATCH --account=gfdl_o
#SBATCH --partition=analysis
#SBATCH --constraint=bigmem

#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=512G
#SBATCH --time=24:00:00

#SBATCH -o dumps/slurm/umap_log_%j.out
#SBATCH -e dumps/slurm/umap_log_%j.err

set -euo pipefail

########################
# Environment
########################
module purge
module load conda
conda activate /work/lnd/ODRI/CONDA/conda_envs/nemi_env


###############################################################################
# Arguments 
###############################################################################

SCRIPT_DIR="/home/Laique.Djeutchouang/DEVs/BV-Regimes/NEMI/seaLevelRegimes/step_1"
PYTHON_FILE="${SCRIPT_DIR}/robust_embedding.py"

INPUT_FILE=$1
UMAP_MD=$2
UMAP_NN=$3
OUTFILE=$4
MEMBER=$5

###############################################################################
# Skip completed work (idempotent)
###############################################################################

if [[ -f "${OUTFILE}" ]]; then
    echo
    echo
    echo "Skipping the following run: ENS=${MEMBER}, UMAP_MD=${UMAP_MD}, UMAP_NN=${UMAP_NN}"
    echo "The output for this run already exists."
    echo
    echo
    exit 0
fi

###############################################################################
# Run without Profiling
###############################################################################

python -u "${PYTHON_FILE}" \
    "${INPUT_FILE}" \
    "${UMAP_MD}" \
    "${UMAP_NN}" \
    "${OUTFILE}" \
    "${MEMBER}"