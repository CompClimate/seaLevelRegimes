#!/bin/bash
#SBATCH --job-name=UMAP_ARRAY
#SBATCH --account=gfdl_o
#SBATCH --partition=analysis
#SBATCH --constraint=bigmem

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1000G
#SBATCH --time=24:00:00

# 20 ensembles × 25 UMAP combos = 500 jobs
#SBATCH --array=0-499%2 # Max 2 concurrent jobs

#SBATCH -o dumps/slurm/umap_log_%A_%a.out
#SBATCH -e dumps/slurm/umap_log_%A_%a.err

###############################################################################
# Safety & Environment Settings
###############################################################################
set -euo pipefail # Exit on error, undefined variable, or error in a pipeline

module purge
module load conda
conda activate /work/lnd/ODRI/CONDA/conda_envs/nemi_env

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

###############################################################################
# Parameter Space Settings
###############################################################################

# 20 ensembles for parameter fine-tuning based on entropy
ENSEMBLES=20

# UMAP Parameters per ensemble member: min_dist and n_neighbors - 5 values each - 25 combos
UMAP_MDS=(0.1 0.3 0.5 0.7 0.9)
UMAP_NNS=(5 10 50 100 200)

NUM_MD=${#UMAP_MDS[@]}
NUM_NN=${#UMAP_NNS[@]}
NUM_UMAP=$((NUM_MD * NUM_NN))

IDX=${SLURM_ARRAY_TASK_ID}

MEMBER=$(( IDX / NUM_UMAP + 1 ))
UMAP_IDX=$(( IDX % NUM_UMAP ))
MD_IDX=$(( UMAP_IDX / NUM_NN ))
NN_IDX=$(( UMAP_IDX % NUM_NN ))

UMAP_MD=${UMAP_MDS[$MD_IDX]}
UMAP_NN=${UMAP_NNS[$NN_IDX]}

###############################################################################
# Paths & Filenames Settings
###############################################################################
DATA_RES="p25"
DATA_FIELD="dynamics"

BASE_DIR="/work/lnd/CM4X/NEMI/CM4X-${DATA_RES}"
INPUT_FILE="${BASE_DIR}/inputs/global_BVB_mclim_fields_scaled.parquet"

SCRIPT_DIR="/home/Laique.Djeutchouang/DEVs/BV-Regimes/NEMI/seaLevelRegimes/step_1"
PYTHON_FILE="${SCRIPT_DIR}/arrayed_embedding.py"

OUTDIR="${BASE_DIR}/outputs/${DATA_FIELD}/embeddings"
CHKDIR="${OUTDIR}/checkpoints"
LOGDIR="${OUTDIR}/logs"

mkdir -p "${OUTDIR}" "${CHKDIR}" "${LOGDIR}"

SUFFIX=$(printf "emb_%02dth_ensemble_md_%s_nn_%s" "$MEMBER" "$UMAP_MD" "$UMAP_NN")
OUTFILE="${OUTDIR}/${SUFFIX}.npy"
CHKFILE="${CHKDIR}/${SUFFIX}.ckpt"

PROFILE_TMP="${LOGDIR}/profile_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
PROFILE_CSV="${LOGDIR}/embedding_profiling_summary.csv"

###############################################################################
# Skip Completed Work (idempotent)
###############################################################################

if [[ -f "${OUTFILE}" ]]; then
    echo
    echo
    echo "Skipping the following run: ENS=${MEMBER}, UMAP_MD=${UMAP_MD}, UMAP_NN=${UMAP_NN}"
    echo "The output for this run already exists."
    echo
    echo
    exit 0
fi

###############################################################################
# Run Jobs + Profiling Each
###############################################################################
START_TIME=$(date +%s)

/usr/bin/time -v python -u "${PYTHON_FILE}" \
    "${INPUT_FILE}" \
    "${UMAP_MD}" \
    "${UMAP_NN}" \
    "${OUTFILE}" \
    --checkpoint "${CHKFILE}" \
    1> "${PROFILE_TMP}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

###############################################################################
# Extract Metrics
###############################################################################
MAX_RSS=$(grep "Maximum resident set size" "${PROFILE_TMP}" | awk '{print $6}')
USER_TIME=$(grep "User time" "${PROFILE_TMP}" | awk '{print $4}')
SYS_TIME=$(grep "System time" "${PROFILE_TMP}" | awk '{print $4}')

###############################################################################
# Append to Summary CSV (safe for job arrays)
###############################################################################
if [[ ! -f "${PROFILE_CSV}" ]]; then
    echo "job_id,array_id,member,min_dist,n_neighbors,cpus,wall_time_s,max_rss_kb,user_time_s,sys_time_s" \
        >> "${PROFILE_CSV}"
fi

echo "${SLURM_JOB_ID},${SLURM_ARRAY_TASK_ID},${MEMBER},${UMAP_MD},${UMAP_NN},${SLURM_CPUS_PER_TASK},${ELAPSED},${MAX_RSS},${USER_TIME},${SYS_TIME}" \
    >> "${PROFILE_CSV}"

rm -f "${PROFILE_TMP}"
