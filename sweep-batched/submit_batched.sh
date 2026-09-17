#!/bin/bash
###############################################################################
# Paths
###############################################################################
DATA_RES="p125"
# DATA_FIELD="statics"
DATA_FIELD="dynamics"
MD=0.0003
NN=15

BASE_DIR="/work/lnd/CM4X/NEMI/Farm"
INPUT_FILE="${BASE_DIR}/CM4X-${DATA_RES}/outputs/${DATA_FIELD}/embeddings/emb_01th_ensemble_md_${MD}_nn_${NN}.npy"
OUTPUT_DIR="${BASE_DIR}/DBSCAN/SWEEP/BATCHED/${DATA_FIELD}/${DATA_RES}"


###############################################################################
# Submit SLURM Jobs
###############################################################################
echo
echo "======================================================================================================"
echo "Submitting DBSCAN SWEEP BATCHED Jobs to SLURM ..."
sbatch --job-name=B:${DATA_RES}:${DATA_FIELD}:${MD}:${NN} sweep_batched.sh \
    "${INPUT_FILE}" "${OUTPUT_DIR}" "${MD}" "${NN}"

echo "Running DBSCAN SWEEP BATCHED | Model: CM4X-${DATA_RES} - ${DATA_FIELD} | UMAP: MD=${MD} - NN=${NN} ..."
echo "======================================================================================================"
echo
# End of script

