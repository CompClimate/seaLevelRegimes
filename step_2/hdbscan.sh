#!/bin/bash
#SBATCH --job-name=GPU-HDBSCAN-SWEEP
#SBATCH --account=maikesgrp
#SBATCH --partition=gpu-h100-h
##SBATCH --partition=high

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00

#SBATCH -o dumps/hdbscan/slog_%j.out
#SBATCH -e dumps/hdbscan/slog_%j.err

########################
# Environment
########################
echo
echo "Loading modules and activating conda environment ..."
module purge
module load conda

# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate /quobyte/maikesgrp/laique/CONDA/conda_envs/nemi_env

eval "$(mamba shell hook --shell bash)"
mamba activate /quobyte/maikesgrp/laique/CONDA/nemi-gpu

echo "Environment ready."
echo

########################
# GPU diagnostics
########################
echo "========================= GPU visibility =========================="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
           --format=csv,noheader 2>/dev/null \
    || echo "WARNING: nvidia-smi failed – GPU may not be accessible"
python -c "import cupy; print('cupy device:', cupy.cuda.Device().id)" 2>/dev/null \
    || echo "WARNING: cupy not importable – cuML will not use GPU"
echo "==================================================================="
echo


########################
# Notes
# ─────────────────────────────────────────────────────────────────────────────
# Parameter guide
#  --min-cluster         minimum cluster size (primary HDBSCAN knob)
#  --min-samples         minimum samples (controls noise conservativeness)
#  --cluster-selection-method  eom (default, larger clusters)
#                               leaf (more, smaller clusters)
#  --cluster-selection-epsilon post-clustering merge threshold (0 = off)
# ─────────────────────────────────────────────────────────────────────────────
# Selection is ranked by composite_score = eff_coverage_hhi × max(0, rel_val)
# when relative_validity_ is available (sklearn/hdbscan backends).
# Falls back to eff_coverage_hhi when using the cuML GPU backend.
########################

# $1 is a space-separated list of ensemble member indices, e.g. "1 2 3 ... 20"
read -ra ENSEMBLES <<< "$1"
MD=$2
NN=$3
MIN_CLUSTER=$4
MIN_SAMPLES=$5
INPUT_DIR=$6
OUTPUT_DIR=$7

for MEMBER in "${ENSEMBLES[@]}"
do
    # Input embedding produced by the UMAP step
    SUFFIX=$(printf "emb_%02dth_ensemble_md_%s_nn_%s" "$MEMBER" "$MD" "$NN")
    INPUT_FILE="${INPUT_DIR}/${SUFFIX}.npy"

    # Output file produced by hdbscan_clustering.py
    OUTPUT_FILE="${OUTPUT_DIR}/mcs${MIN_CLUSTER}_ms${MIN_SAMPLES}_eom/emb_$(printf '%02d' "$MEMBER")th_ensemble_md${MD}_nn${NN}.npz"

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
        echo "Skipping HDBSCAN on Embedding: ${SUFFIX} (output already exists)"
        echo
        continue
    fi

    echo
    echo "Running HDBSCAN clustering on Embedding: ${SUFFIX} ..."
    python -u hdbscan_clustering.py \
        "${INPUT_FILE}" \
        --min-cluster "${MIN_CLUSTER}" \
        --min-samples "${MIN_SAMPLES}" \
        --umap-md "${MD}" --umap-nn "${NN}" \
        --ens-member "${MEMBER}" --output-dir "${OUTPUT_DIR}"
done

echo
# End of script
