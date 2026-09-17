#!/bin/bash

###############################################################################
# Paths
###############################################################################
DATA_RES="p125"
DATA_FIELD="statics"
# DATA_FIELD="dynamics"

BASE_DIR="/group/maikesgrp/laique/NOAA/NEMI"
INPUT_DIR="${BASE_DIR}/CM4X-${DATA_RES}/outputs/${DATA_FIELD}/embeddings"
OUTPUT_DIR="${BASE_DIR}/HDBSCAN/${DATA_FIELD}/${DATA_RES}/clusterings"

###############################################################################
# SLURM job configuration
###############################################################################
ENSEMBLES=(1 2 3 4 5) #(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
MD=0.1
NN=200
MIN_CLUSTER=500
MIN_SAMPLES=500

###############################################################################
# Submit SLURM Jobs
###############################################################################
echo
echo "========================================================================="
echo "Submitting HDBSCAN Sweep Jobs to SLURM ..."
sbatch --job-name=HDB:${DATA_RES} hdbscan_clustering.sh \
    "${ENSEMBLES[*]}" "${MD}" "${NN}" \
    "${MIN_CLUSTER}" "${MIN_SAMPLES}" \
    "${INPUT_DIR}" "${OUTPUT_DIR}"

echo "Running HDBSCAN SWEEP: Grid Search | ${DATA_RES} - ${DATA_FIELD} ..."
echo "========================================================================="
echo
# End of script
