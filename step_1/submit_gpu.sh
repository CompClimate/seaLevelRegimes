#!/bin/bash

# ===============================================================
# Hyperparameter Configuration
# ===============================================================
ENSEMBLES=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
MDS=(0.005)
NNS=(5)


###############################################################################
# Paths
###############################################################################
DATA_RES="p125"
BASE_DIR="/group/maikesgrp/laique/NOAA/NEMI/CM4X-${DATA_RES}"
SCRIPT_DIR="/home/djeutsch/Projects/seaLevelRegimes/step_1"

DATA_FIELD="statics"
INPUT_FILE="${BASE_DIR}/inputs/global_BVB_tmean_fields_scaled.parquet"

# DATA_FIELD="dynamics"
# INPUT_FILE="${BASE_DIR}/inputs/global_BVB_mclim_fields_scaled.parquet"

OUTDIR="${BASE_DIR}/outputs/${DATA_FIELD}/embeddings"
mkdir -p "${OUTDIR}"


# =============================================================================
# Submit Jobs
# =============================================================================
for MD in "${MDS[@]}"; do
    for NN in "${NNS[@]}"; do
        for MEMBER in "${ENSEMBLES[@]}"; do
            SUFFIX=$(printf "emb_%02dth_ensemble_md_%s_nn_%s" "$MEMBER" "$MD" "$NN")
            OUTFILE="${OUTDIR}/${SUFFIX}.npy"
            if [[ -f "${OUTFILE}" ]]; then
                echo
                echo "====================================================================================="
                echo "Skipping embedding | CM4X: ${DATA_RES} - ${DATA_FIELD} | UMAP: MD=${MD} - NN=${NN} ... (already exists)"
                echo "====================================================================================="
                echo
                continue
            fi
            echo
            echo "========================================================================================="
            echo "Running GPU Embedding: CM4X: ${DATA_RES} - ${DATA_FIELD} | UMAP: MD=${MD} - NN=${NN} ..."
            sbatch --job-name=E${MEMBER}:${DATA_RES}:MD${MD}:NN${NN} \
                "${SCRIPT_DIR}/gpu_embedding.sh" \
                "${INPUT_FILE}" \
                "${MD}" \
                "${NN}" \
                "${MEMBER}" \
                "${OUTFILE}"
            echo "========================================================================================="
            echo
        done
    done
done
