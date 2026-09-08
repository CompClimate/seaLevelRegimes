#!/bin/bash
# =============================================================================
# Submit the BV(B) sea-level regime NN pipeline (pipeline.sh -> pipeline.py).
#
# Job-resource options are consumed here and turned into sbatch flags; every
# other option is forwarded verbatim to `pipeline.py main()`. Run
#   ./submit.sh --help              for the job options and the defaults
#   python pipeline.py --help       for the full pipeline option reference
#
# Examples
#   ./submit.sh                                     # defaults: train + predict
#   ./submit.sh --epochs 200 --hidden 512,256,128,64,32
#   ./submit.sh --gpus 1 --batch-size 32768 --class-weights curriculum
#   ./submit.sh --mode predict --checkpoint /path/model.pt --predict-years 2012 2014
#   ./submit.sh --dry-run --train-years 2005 2011    # print the sbatch command
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------------------------------------------------------
# Defaults - edit here, or override on the command line
# -----------------------------------------------------------------------------
# Data / experiment
BASE_DIR="${SLVP_BASE_DIR:-/group/maikesgrp/laique/PPAN/CM4X/NN4X}"
INPUT="${BASE_DIR}/inputs/global_NN4X_p25_monthly_features_nc15.zarr"
OUTDIR="${BASE_DIR}/outputs/nn"

# Slurm resources
ACCOUNT="maikesgrp"
PARTITION="gpu-h100-h"                 # resolved below: 'analysis' (CPU) or 'gpu'
TIME_LIMIT="24:00:00"
MEMORY="350G"
CPUS="32"
GPUS="1"                     # >0 switches to the GPU partition
GPU_TYPE="h100"
JOB_NAME=""                  # derived from --mode and --tag when empty
LOGDIR="${SCRIPT_DIR}/dumps/nn"
CONSTRAINT=""                # Slurm feature constraint, e.g. 'bigmem'
# The 'analysis' partition is heterogeneous: an001/an002 are pre-AVX (2010)
# Xeons on which prebuilt PyTorch wheels die with SIGILL. Exclude them by
# default; pass '--exclude ""' to allow them.
EXCLUDE_NODES=""

# Runtime environment (must provide pytorch, xarray, zarr, scikit-learn)
CONDA_ENV="${SLVP_CONDA_ENV:-/quobyte/maikesgrp/laique/CONDA/conda_envs/nemi_env}"

DRY_RUN=0

# pipeline.py options: defaults here, overridden by matching command-line flags.
declare -A OPT=(
    [--input]="${INPUT}"
    [--outdir]="${OUTDIR}"
    [--mode]="train-predict"
    [--train-years]="2005 2011"
    [--predict-years]=""
    [--predict-input]=""
    [--features]=""
    [--label-var]=""
    [--time-stride]=""
    [--n-regimes]="15"
    [--hidden]="256,128,64,32,16"
    [--dropout]=""
    [--epochs]="150"
    [--batch-size]="16384"
    [--rare-regimes]=""
    [--overwrite]=""
    [--verbose]=""
    [--lr]=""
    [--weight-decay]=""
    [--train-frac]=""
    [--patience]=""
    [--min-delta]=""
    [--lambda-entropy]=""
    [--sched-factor]=""
    [--sched-patience]=""
    [--sched-threshold]=""
    [--sched-cooldown]=""
    [--min-lr]=""
    [--class-weights]="curriculum"
    [--curriculum-warmup]="25"
    [--entropy-unit]=""
    [--predict-batch-size]=""
    [--predict-time-chunk]=""
    [--tag]=""
    [--pred-format]=""
    [--checkpoint]=""
    [--device]=""
    [--data-on-device]=""
    [--seed]=""
)
declare -a FLAGS=()      # valueless pipeline.py options
declare -a EXTRA=()      # anything after a bare '--'

usage() {
    cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [job options] [pipeline.py options] [-- extra args]

Job options (consumed by this script):
  --account NAME        Slurm account                     [${ACCOUNT}]
  --partition NAME      Slurm partition                   [analysis, or gpu when --gpus > 0]
  --time HH:MM:SS       Wall-clock limit                  [${TIME_LIMIT}]
  --mem SIZE            Memory per node                   [${MEMORY}]
  --cpus N              CPUs per task                     [${CPUS}]
  --gpus N              GPUs (0 = CPU-only run)           [${GPUS}]
  --gpu-type NAME       GRES GPU type                     [${GPU_TYPE}]
  --job-name NAME       Slurm job name                    [derived from --mode/--tag]
  --logdir DIR          Directory for the .out/.err logs  [${LOGDIR}]
  --constraint FEAT     Slurm feature constraint          [none]
  --exclude NODES       Nodes to avoid ('' to allow all)  [${EXCLUDE_NODES}]
  --env PATH            Conda environment to activate     [${CONDA_ENV}]
  --base-dir DIR        Data root for the default paths   [${BASE_DIR}]
  --dry-run             Print the sbatch command instead of submitting
  -h, --help            Show this message

Pipeline options (forwarded to pipeline.py; see 'python pipeline.py --help'):
  -i, --input PATH            [${OPT[--input]}]
  -o, --outdir DIR            [${OPT[--outdir]}]
      --mode MODE             train | predict | train-predict  [${OPT[--mode]}]
      --train-years S E       [${OPT[--train-years]}]
      --predict-years S E     [whole record]
      --predict-input PATH    [same as --input]
      --features V1,V2,...    [the nine BVB terms]
      --label-var NAME        [bvb_regime]
      --n-regimes N           [${OPT[--n-regimes]}]
      --hidden H1,H2,...      [${OPT[--hidden]}]
  -e, --epochs N              [${OPT[--epochs]}]
  -b, --batch-size N          [${OPT[--batch-size]}]
      --lr, --weight-decay, --train-frac, --dropout, --time-stride
      --patience, --min-delta, --lambda-entropy, --min-lr
      --sched-factor, --sched-patience, --sched-threshold, --sched-cooldown
      --class-weights {balanced|curriculum|none}, --curriculum-warmup
      --entropy-unit {fraction|percent}, --predict-batch-size, --predict-time-chunk
      --tag, --pred-format {zarr|nc}, --checkpoint, --device, --data-on-device, --seed
      --rare-regimes | --no-rare-regimes, --overwrite, -v/--verbose   (flags)
EOF
}

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        # ---- job options -----------------------------------------------------
        --account)      ACCOUNT="$2";    shift 2 ;;
        --partition)    PARTITION="$2";  shift 2 ;;
        --time)         TIME_LIMIT="$2"; shift 2 ;;
        --mem)          MEMORY="$2";     shift 2 ;;
        --cpus)         CPUS="$2";       shift 2 ;;
        --gpus)         GPUS="$2";       shift 2 ;;
        --gpu-type)     GPU_TYPE="$2";   shift 2 ;;
        --job-name)     JOB_NAME="$2";   shift 2 ;;
        --logdir)       LOGDIR="$2";     shift 2 ;;
        --constraint)   CONSTRAINT="$2"; shift 2 ;;
        --exclude)      EXCLUDE_NODES="$2"; shift 2 ;;
        --env)          CONDA_ENV="$2";  shift 2 ;;
        --base-dir)
            BASE_DIR="$2"
            OPT[--input]="${BASE_DIR}/inputs/monthly_bvb_nn_features_num_labels.zarr"
            OPT[--outdir]="${BASE_DIR}/outputs/nn"
            shift 2 ;;
        --dry-run)      DRY_RUN=1;       shift ;;
        -h|--help)      usage; exit 0 ;;

        # ---- pass-through separator (must precede the --* fallback) ----------
        --)             shift; EXTRA+=("$@"); break ;;

        # ---- pipeline.py flags (no value) ------------------------------------
        --rare-regimes|--no-rare-regimes|--overwrite|--verbose)
                        FLAGS+=("$1");   shift ;;
        -v)             FLAGS+=("--verbose"); shift ;;

        # ---- pipeline.py options taking two values ---------------------------
        --train-years|--predict-years)
            if [[ $# -lt 3 || "$2" == -* || "$3" == -* ]]; then
                echo "ERROR: option '$1' expects START and END years, e.g. '$1 2005 2011'." >&2
                exit 2
            fi
            OPT["$1"]="$2 $3"; shift 3 ;;

        # ---- pipeline.py short options ---------------------------------------
        -i)             OPT[--input]="$2";      shift 2 ;;
        -o)             OPT[--outdir]="$2";     shift 2 ;;
        -e)             OPT[--epochs]="$2";     shift 2 ;;
        -b)             OPT[--batch-size]="$2"; shift 2 ;;

        # ---- any other pipeline.py long option taking one value --------------
        --*)
            if [[ $# -lt 2 || "$2" == -* ]]; then
                echo "ERROR: option '$1' expects a value (use '--' to pass raw flags)." >&2
                exit 2
            fi
            OPT["$1"]="$2"; shift 2 ;;

        *)
            echo "ERROR: unexpected argument '$1'." >&2
            usage >&2
            exit 2 ;;
    esac
done

# -----------------------------------------------------------------------------
# Resolve the job configuration
# -----------------------------------------------------------------------------
GRES=""
if (( GPUS > 0 )); then
    PARTITION="${PARTITION:-gpu}"
    GRES="gpu:${GPU_TYPE}:${GPUS}"
else
    PARTITION="${PARTITION:-analysis}"
fi

if [[ -z "${JOB_NAME}" ]]; then
    JOB_NAME="BVB:NN:${OPT[--mode]}"
    if [[ -n "${OPT[--tag]}" ]]; then
        JOB_NAME="${JOB_NAME}:${OPT[--tag]}"
    fi
fi

PIPELINE_SH="${SCRIPT_DIR}/pipeline.sh"
[[ -f "${PIPELINE_SH}" ]] || { echo "ERROR: ${PIPELINE_SH} not found." >&2; exit 1; }

# Fail early on the obvious mistakes rather than after the job is queued.
if [[ ! -e "${OPT[--input]}" ]]; then
    echo "ERROR: input dataset not found: ${OPT[--input]}" >&2
    exit 1
fi
if [[ "${OPT[--mode]}" == "predict" && -z "${OPT[--checkpoint]}" && -z "${OPT[--tag]}" ]]; then
    echo "WARNING: --mode predict without --checkpoint or --tag; pipeline.py will look" >&2
    echo "         for a checkpoint under the derived default name." >&2
fi

mkdir -p "${LOGDIR}"
mkdir -p "${OPT[--outdir]}"

# -----------------------------------------------------------------------------
# Assemble the pipeline.py argument list
# -----------------------------------------------------------------------------
declare -a PIPE_ARGS=()
for key in "${!OPT[@]}"; do
    value="${OPT[$key]}"
    [[ -z "${value}" ]] && continue
    if [[ "${key}" == "--train-years" || "${key}" == "--predict-years" ]]; then
        read -r year_start year_end <<< "${value}"
        PIPE_ARGS+=("${key}" "${year_start}" "${year_end}")
    else
        PIPE_ARGS+=("${key}" "${value}")
    fi
done
if (( ${#FLAGS[@]} )); then PIPE_ARGS+=("${FLAGS[@]}"); fi
if (( ${#EXTRA[@]} )); then PIPE_ARGS+=("${EXTRA[@]}"); fi

# -----------------------------------------------------------------------------
# Assemble the sbatch command
# -----------------------------------------------------------------------------
declare -a SBATCH_ARGS=(
    --job-name="${JOB_NAME}"
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --nodes=1
    --ntasks=1
    --cpus-per-task="${CPUS}"
    --mem="${MEMORY}"
    --time="${TIME_LIMIT}"
    --output="${LOGDIR}/nn_%j.out"
    --error="${LOGDIR}/nn_%j.err"
    --export="ALL,SLVP_NN_DIR=${SCRIPT_DIR},SLVP_CONDA_ENV=${CONDA_ENV},SLVP_BASE_DIR=${BASE_DIR}"
)
if [[ -n "${GRES}" ]]; then
    SBATCH_ARGS+=(--gres="${GRES}")
fi
if [[ -n "${CONSTRAINT}" ]]; then
    SBATCH_ARGS+=(--constraint="${CONSTRAINT}")
fi
if [[ -n "${EXCLUDE_NODES}" ]]; then
    SBATCH_ARGS+=(--exclude="${EXCLUDE_NODES}")
fi

echo
echo "======================================================================="
echo "Submitting the BV(B) regime NN pipeline"
echo "-----------------------------------------------------------------------"
echo "  job name     : ${JOB_NAME}"
echo "  partition    : ${PARTITION} (account ${ACCOUNT})"
echo "  resources    : ${CPUS} cpus, ${MEMORY} mem, ${GPUS} gpu(s), ${TIME_LIMIT}"
echo "  excluding    : ${EXCLUDE_NODES:-<none>}"
echo "  conda env    : ${CONDA_ENV}"
echo "  logs         : ${LOGDIR}/nn_<jobid>.{out,err}"
echo "  mode         : ${OPT[--mode]}"
echo "  input        : ${OPT[--input]}"
echo "  outdir       : ${OPT[--outdir]}"
echo "  pipeline args: ${PIPE_ARGS[*]}"
echo "======================================================================="
echo

if (( DRY_RUN )); then
    echo "[dry-run] sbatch ${SBATCH_ARGS[*]} ${PIPELINE_SH} ${PIPE_ARGS[*]}"
    exit 0
fi

sbatch "${SBATCH_ARGS[@]}" "${PIPELINE_SH}" "${PIPE_ARGS[@]}"
