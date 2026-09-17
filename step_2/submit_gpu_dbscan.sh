#!/bin/bash

###############################################################################
# Paths
###############################################################################
DATA_RES="p25"
# DATA_FIELD="statics"
DATA_FIELD="dynamics"

BASE_DIR="/group/maikesgrp/laique/NOAA/NEMI"
INPUT_DIR="${BASE_DIR}/CM4X-${DATA_RES}/outputs/${DATA_FIELD}/embeddings"
OUTPUT_DIR="${BASE_DIR}/DBSCAN/${DATA_FIELD}/${DATA_RES}/clusterings"

mkdir -p "${OUTPUT_DIR}"

###############################################################################
# SLURM job configuration
###############################################################################
ENSEMBLES=(1) # 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
MD=0.001
NN=10
EPSILON=0.005
MIN_SAMPLES=20

###############################################################################
# Submit SLURM Jobs
###############################################################################
echo
echo "==============================================================================================================="
echo "Submitting a DBSCAN Job of 20-Member Ensemble to SLURM with the Following Configuration..."
sbatch --job-name=E:D:${DATA_RES}:MD${MD}:NN${NN} gpu_dbscan.sh \
    "${ENSEMBLES[*]}" "${MD}" "${NN}" \
    "${EPSILON}" "${MIN_SAMPLES}" \
    "${INPUT_DIR}" "${OUTPUT_DIR}"

echo "DBSCAN: EPS=${EPSILON} - MS=${MIN_SAMPLES} | CM4X: ${DATA_RES} - ${DATA_FIELD} | UMAP: MD=${MD} - NN=${NN} ..."
echo "==============================================================================================================="
echo
# End of script
