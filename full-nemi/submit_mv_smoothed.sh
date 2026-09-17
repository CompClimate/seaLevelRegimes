#!/bin/bash

###############################################################################
# Paths
###############################################################################
DATA_RES="p25"
DATA_FIELD="statics"
# DATA_FIELD="dynamics"

BASE_DIR="/work/lnd/CM4X/NEMI/Farm"
PPAN_DIR="/work/lnd/CM4X"


###############################################################################
# SLURM job configuration
###############################################################################
MEMBER_SIZE=20
MD=0.001
NN=5
EPSILON=0.006
MIN_SAMPLES=40
TOP_K_CLUSTERS=200

GRID_ZARR="${PPAN_DIR}/BVB/CM4X-${DATA_RES}_BVB_and_tracers_geocoords_2005_2014_time_mean.zarr"
DATA_DIR="${BASE_DIR}/DBSCAN/${DATA_FIELD}/${DATA_RES}/clusterings/eps${EPSILON}_ms${MIN_SAMPLES}"
OUTPUT_FILE="${DATA_DIR}/majority_vote_md${MD}_nn${NN}_eps${EPSILON}_ms${MIN_SAMPLES}.npz"


###############################################################################
# Submit SLURM Jobs
###############################################################################
echo
echo "==============================================================================================================="
echo "Submitting a NEMI-DBSCAN Job with ${MEMBER_SIZE}-Member Ensemble to SLURM from the Following Configuration..."
sbatch --job-name=NEMI:${DATA_RES}:MD${MD}:NN${NN} majority_vote_smoothed.sh \
    "${MEMBER_SIZE}" "${MD}" "${NN}" \
    "${EPSILON}" "${MIN_SAMPLES}" \
    "${DATA_DIR}" "${OUTPUT_FILE}" \
    "${TOP_K_CLUSTERS}" "${GRID_ZARR}"

echo "DBSCAN: EPS=${EPSILON} - MS=${MIN_SAMPLES} | CM4X: ${DATA_RES} - ${DATA_FIELD} | UMAP: MD=${MD} - NN=${NN} ..."
echo "==============================================================================================================="
echo
# End of script
