#!/bin/bash
###############################################################################
# CONFIGURATION
###############################################################################
DATA_RES="p25"
# DATA_FIELD="statics"
DATA_FIELD="dynamics"
ENSEMBLES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
MD=0.001
NN=10
EPSILON=0.003
MIN_SAMPLES=50

BASE_DIR="/work/lnd/CM4X/NEMI/Farm"
INPUT_DIR="${BASE_DIR}/CM4X-${DATA_RES}/outputs/${DATA_FIELD}/embeddings"
OUTPUT_DIR="${BASE_DIR}/DBSCAN/${DATA_FIELD}/${DATA_RES}/clusterings"

mkdir -p "${OUTPUT_DIR}"

###############################################################################
# Submit SLURM Jobs
###############################################################################
echo
for MEMBER in "${ENSEMBLES[@]}"; do
    echo "==============================================================================================================="
    echo "Submitting a DBSCAN Batched Job of ${MEMBER}-Member Ensemble to SLURM with the Following Configuration ..."
    MEMBER_SET=("${MEMBER}")
    sbatch --job-name=DB:${MEMBER}:${DATA_RES}:MD${MD}:NN${NN} dbs_batched.sh \
        "${MEMBER_SET[*]}" "${MD}" "${NN}" \
        "${EPSILON}" "${MIN_SAMPLES}" \
        "${INPUT_DIR}" "${OUTPUT_DIR}"

    echo "DBSCAN: EPS=${EPSILON} - MS=${MIN_SAMPLES} | CM4X: ${DATA_RES} - ${DATA_FIELD} | UMAP: MD=${MD} - NN=${NN} ..."
    echo "==============================================================================================================="
    echo
done
# End of script



