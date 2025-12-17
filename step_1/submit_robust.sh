#!/bin/bash

# ==========================
# Hyperparameter Configuration
# ==========================
ENSEMBLES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
UMAP_MDS=(0.1 0.3 0.5 0.7 0.9)
UMAP_NNS=(5 10 50 100 200)


###############################################################################
# Paths
###############################################################################
DATA_RES="p25"
DATA_FIELD="dynamics"

BASE_DIR="/work/lnd/CM4X/NEMI/CM4X-${DATA_RES}"
INPUT_FILE="${BASE_DIR}/inputs/global_BVB_mclim_fields_scaled.parquet"

OUTDIR="${BASE_DIR}/outputs/${DATA_FIELD}/embeddings"

mkdir -p "${OUTDIR}"

# ==========================
# Submit Jobs
# ==========================
for UMAP_MD in "${UMAP_MDS[@]}"; do
    for UMAP_NN in "${UMAP_NNS[@]}"; do
        for MEMBER in "${ENSEMBLES[@]}"; do
            SUFFIX=$(printf "emb_%02dth_ensemble_md_%s_nn_%s" "$MEMBER" "$UMAP_MD" "$UMAP_NN")
            OUTFILE="${OUTDIR}/${SUFFIX}.npy"

            echo "Submitting: ENS=${MEMBER}, min_dist=${UMAP_MD}, n_neighbors=${UMAP_NN}"
            sbatch \
                --job-name=UMAP_ENS:${MEMBER}_MD:${UMAP_MD}_NN:${UMAP_NN} \
                run_robust.sh \
                "${INPUT_FILE}" \
                "${UMAP_MD}" \
                "${UMAP_NN}" \
                "${OUTFILE}" \
                "${MEMBER}"
        done
    done
done
