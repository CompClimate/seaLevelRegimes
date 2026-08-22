#!/bin/bash
#SBATCH --job-name=UMAP:EMB
##SBATCH --account=maikesgrp 
##SBATCH --partition=gpu-h100-h 
#SBATCH --account=publicgrp
#SBATCH --partition=low

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24

##SBATCH --gres=gpu:h100:1
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=512G

#SBATCH --time=12:00:00

#SBATCH -o dumps/gpu/emb_log_%j.out
#SBATCH -e dumps/gpu/emb_log_%j.err


########################
# Environment
########################
echo
echo "Loading modules and activating conda environment ..."
module purge
module load conda

eval "$(mamba shell hook --shell bash)"
mamba activate /quobyte/maikesgrp/laique/CONDA/nemi-gpu-25

echo "Environment fully loaded and ready."
echo

########################
# GPU diagnostics
########################
echo "========================= GPU visibility =========================="
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-<not set by SLURM>}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
           --format=csv,noheader 2>/dev/null \
    || echo "WARNING: nvidia-smi failed – GPU may not be accessible"
python -c "
import cupy, sys
n = cupy.cuda.runtime.getDeviceCount()
if n == 0:
    print('WARNING: cupy sees 0 GPU devices', file=sys.stderr)
    sys.exit(1)
cupy.cuda.Device(0).use()
props = cupy.cuda.runtime.getDeviceProperties(0)
print(f'cupy OK  – device 0: {props[\"name\"].decode()} ({cupy.cuda.Device(0).mem_info[1] // 1024**2} MB free)')
" 2>&1 || echo "WARNING: cupy probe failed – cuML will fall back to CPU"
echo "==================================================================="
echo


###############################################################################
# Arguments
###############################################################################
SCRIPT_DIR="/home/djeutsch/Projects/seaLevelRegimes/step_1"
PYTHON_FILE="${SCRIPT_DIR}/gpu_embedding.py"

INPUT_FILE=$1
MD=$2
NN=$3
MEMBER=$4
OUTFILE=$5

###############################################################################
# Run
###############################################################################
if [[ ! -f "${OUTPUT_FILE}" ]]; then
    python -u "${PYTHON_FILE}" \
        "${INPUT_FILE}" \
        "${MD}" \
        "${NN}" \
        "${MEMBER}" \
        "${OUTFILE}"
else # Skip completed work (idempotent)
    echo
    echo "========================================================================="
    echo "Skipping embedding: MD=${MD}, NN=${NN} (already exists)"
    echo "========================================================================="
    echo
    exit 0
fi