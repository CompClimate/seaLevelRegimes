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

#SBATCH -o dumps/dbscan/batched/clim_slog_%j.out
#SBATCH -e dumps/dbscan/batched/clim_slog_%j.err

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

if [ "$#" -ne 7 ]; then
    echo "Usage: sbatch dbs_batched.sh <ensembles> <umap_md> <umap_nn> <epsilon> <min_samples> <input_dir> <output_dir>" >&2
    exit 1
fi


# $1 is a space-separated list of ensemble member indices, e.g. "1 2 3 ... 20"
read -ra ENSEMBLES <<< "$1"
MD=$2
NN=$3
EPSILON=$4
MIN_SAMPLES=$5
INPUT_DIR=$6
OUTPUT_DIR=$7

OUTPUT_DIR="${OUTPUT_DIR}/eps${EPSILON}_ms${MIN_SAMPLES}" # _JOB${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_DIR}"

for MEMBER in "${ENSEMBLES[@]}"
do
    # Input embedding produced by the UMAP step
    SUFFIX=$(printf "emb_%02dth_ensemble_md_%s_nn_%s" "$MEMBER" "$MD" "$NN")
    INPUT_FILE="${INPUT_DIR}/${SUFFIX}.npy"

    # Output file produced by cpu_dbscan.py
    OUTPUT_FILE="${OUTPUT_DIR}/emb_$(printf '%02d' "$MEMBER")th_ensemble_md${MD}_nn${NN}.npz"

    if [[ ! -f "${INPUT_FILE}" ]]; then
        echo
        echo "========================================================================="
        echo "WARNING: Input embedding not found, skipping: ${INPUT_FILE}"
        echo
        continue
    fi

    if [[ -f "${OUTPUT_FILE}" ]]; then
        echo
        echo "========================================================================="
        echo "Skipping DBSCAN on Embedding: ${SUFFIX} (output already exists)"
        echo
        continue
    fi

    echo
    echo "Running DBSCAN Clustering on Embedding: ${SUFFIX} ..."
    python -u dbs_batched.py \
        "${INPUT_FILE}" \
        --node-ram-gb "$((SLURM_MEM_PER_NODE / 1024))" \
        --umap-md "${MD}" --umap-nn "${NN}" \
        --eps "${EPSILON}" --min-samples "${MIN_SAMPLES}" \
        --ens-member "${MEMBER}" --output-dir "${OUTPUT_DIR}" --no-normalize
done

echo
# End of script










