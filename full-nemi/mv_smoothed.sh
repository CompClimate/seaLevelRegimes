#!/bin/bash
#SBATCH --job-name=NEMI
#SBATCH --account=gfdl_o
#SBATCH --partition=analysis 
#SBATCH --constraint=bigmem 
##SBATCH --partition=gpu

##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1010G
#SBATCH --time=48:00:00

#SBATCH -o dumps/ppan/slog_%j.out
#SBATCH -e dumps/ppan/slog_%j.err

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


########################
# Arguments
########################

MEMBER_SIZE=$1
MD=$2
NN=$3
EPSILON=$4
MIN_SAMPLES=$5
DATA_DIR=$6
OUTPUT_FILE=$7
TOP_K_CLUSTERS=$8
GRID_ZARR=$9


if [[ -f "${OUTPUT_FILE}" ]]; then
    echo
    echo "========================================================================="
    echo "Skipping: output file already exists:"
    echo "  ${OUTPUT_FILE}"
    echo "========================================================================="
    echo
    exit 0
fi

echo
echo "Running majority vote on DBSCAN ensemble ..."
python -u majority_vote_smoothed.py \
    --eps "${EPSILON}" \
    --min-samples "${MIN_SAMPLES}" \
    --umap-md "${MD}" \
    --umap-nn "${NN}" \
    --member-size "${MEMBER_SIZE}" \
    --data-dir "${DATA_DIR}" \
    --output-file "${OUTPUT_FILE}" \
    --top-k-clusters "${TOP_K_CLUSTERS}" \
    --grid-zarr "${GRID_ZARR}" \
    --smooth-fill-noise
echo
# End of script
