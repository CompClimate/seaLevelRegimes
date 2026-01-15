#!/bin/bash
#SBATCH --job-name=CLUSTERING
#SBATCH --account=gfdl_o
#SBATCH --partition=analysis
## SBATCH --constraint=bigmem

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4 #8
#SBATCH --mem=250G
#SBATCH --time=48:00:00

#SBATCH -o dumps/an/hac_log_%j.out
#SBATCH -e dumps/an/hac_log_%j.err

set -euo pipefail

########################
# Environment
########################
module purge
module load conda
conda activate /work/lnd/ODRI/CONDA/conda_envs/nemi_env


###############################################################################
# Arguments 
###############################################################################

SCRIPT_DIR="/home/Laique.Djeutchouang/DEVs/BV-Regimes/NEMI/seaLevelRegimes/step_2"
PYTHON_FILE="${SCRIPT_DIR}/clustering_on_ppan.py"

INDIR=$1
UMAP_MD=$2
UMAP_NN=$3
MEMBER=$4
HCLUST_N=$5
NC=$6
OUTDIR=$7

###############################################################################
# Run without Profiling
###############################################################################

# Define the input and output files
SUFFIX=$(printf "emb_%02dth_ensemble_md_%s_nn_%s" "$MEMBER" "$UMAP_MD" "$UMAP_NN")
INPUT_FILE="${INDIR}/${SUFFIX}.npy"
OUTPUT_FILE="${OUTDIR}/${SUFFIX}_nc_${NC}_hclustn_${HCLUST_N}.npy"

# Check if the input file exists and the output file does not already exist
if [[ -f "${INPUT_FILE}" ]]; then
    if [[ ! -f "${OUTPUT_FILE}" ]]; then
        python -u "${PYTHON_FILE}" \
            "${INPUT_FILE}" \
            "${HCLUST_N}" \
            "${NC}" \
            "${OUTPUT_FILE}"
    else # Skip completed work (idempotent)
        echo
        echo
        echo "Skipping the following run: ENS=${MEMBER}, UMAP_MD=${UMAP_MD}, UMAP_NN=${UMAP_NN}", "HCLUST_N=${HCLUST_N}, NC=${NC}"
        echo "The output for this run already exists."
        echo
        echo
        exit 0
    fi
else # Skip missing input file
    echo
    echo
    echo "Skipping the following run: ENS=${MEMBER}, UMAP_MD=${UMAP_MD}, UMAP_NN=${UMAP_NN}", "HCLUST_N=${HCLUST_N}, NC=${NC}"
    echo "The required embedding below for this run doesn't exist yet."
    echo "${SUFFIX}"
    echo
    echo
    exit 0
fi




