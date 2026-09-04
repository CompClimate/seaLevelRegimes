#!/bin/bash
# =============================================================================
# Slurm batch script for the BV(B) sea-level regime NN pipeline.
#
# Every argument given to this script is forwarded verbatim to
# `pipeline.py main()`, so the Python CLI stays the single source of truth for
# the options. Resource requests below are defaults; `submit.sh` overrides them
# on the sbatch command line (CLI options win over #SBATCH directives).
#
# Usage (normally through submit.sh):
#   sbatch pipeline.sh --input <features.zarr> --outdir <dir> [pipeline.py options]
#
# Environment variables honoured (exported by submit.sh):
#   SLVP_NN_DIR      directory holding pipeline.py   (default: this repo's nn/)
#   SLVP_CONDA_ENV   conda environment to activate   (must provide pytorch)
#   SLVP_BASE_DIR    data root used by pipeline.py defaults
# =============================================================================

#SBATCH --job-name=BVB:NN
#SBATCH --account=gfdl_o
#SBATCH --partition=analysis

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=12:00:00

#SBATCH -o dumps/nn/nn_log_%j.out
#SBATCH -e dumps/nn/nn_log_%j.err

set -euo pipefail

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
NN_DIR="${SLVP_NN_DIR:-/home/Laique.Djeutchouang/DEVs/SLVP/seaLevelRegimes/nn}"
PYTHON_FILE="${NN_DIR}/pipeline.py"
CONDA_ENV_PATH="${SLVP_CONDA_ENV:-/work/lnd/ODRI/CONDA/conda_envs/nemi_env}"

if [[ ! -f "${PYTHON_FILE}" ]]; then
    echo "ERROR: pipeline.py not found at ${PYTHON_FILE}." >&2
    echo "       Set SLVP_NN_DIR to the directory that contains it." >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    echo "ERROR: no pipeline.py arguments given (at least --input is required)." >&2
    echo "Usage: sbatch pipeline.sh --input <features.zarr> [options]" >&2
    exit 2
fi

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
echo
echo "Loading modules and activating the conda environment ..."
module purge
module load conda
conda activate "${CONDA_ENV_PATH}"
echo "Environment ready: $(python -c 'import sys; print(sys.executable)')"
echo

# Keep the numeric libraries inside the CPU allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

# -----------------------------------------------------------------------------
# Job / hardware diagnostics
# -----------------------------------------------------------------------------
echo "=========================== JOB CONTEXT ==========================="
echo "Job ID          : ${SLURM_JOB_ID:-<interactive>}"
echo "Job name        : ${SLURM_JOB_NAME:-<none>}"
echo "Node(s)         : ${SLURM_JOB_NODELIST:-$(hostname)}"
echo "CPUs per task   : ${SLURM_CPUS_PER_TASK:-<unset>}"
echo "Memory          : ${SLURM_MEM_PER_NODE:-<unset>} MB"
echo "Submitted from  : ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Pipeline        : ${PYTHON_FILE}"
echo "Arguments       : $*"
echo "==================================================================="

if [[ -n "${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}" ]]; then
    echo "============================ GPU CONTEXT =========================="
    echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-<not set by Slurm>}"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \
               --format=csv,noheader 2>/dev/null \
        || echo "WARNING: nvidia-smi failed - the GPU may not be accessible."
    python - <<'PY' 2>&1 || echo "WARNING: torch CUDA probe failed - training will fall back to CPU."
import torch
print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}"
      + (f" | device 0: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
PY
    echo "==================================================================="
fi
echo

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
status=0
python -u "${PYTHON_FILE}" "$@" || status=$?

echo
echo "pipeline.py exited with status ${status}."
exit "${status}"
