#!/bin/bash
#SBATCH --job-name=ENSEMBLE-NN-GPU
#SBATCH --account=maikesgrp
#SBATCH --partition=gpu-h100-h

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=h100:1
#SBATCH --mem=256G
#SBATCH --time=12:00:00

#SBATCH -o dumps/ens_nn_gpu_log_%j.out
#SBATCH -e dumps/ens_nn_gpu_log_%j.err

########################
# Environment
########################
echo
echo "Loading modules and activating conda environment ..."
source /etc/profile.d/modules.sh
module purge
module load conda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /quobyte/maikesgrp/laique/CONDA/conda_envs/nemi_env

echo "Environment ready."
nvidia-smi
echo


########################
# Arguments
########################
echo
echo "Running Ensemble Training and Prediction ..."
cd "$SLURM_SUBMIT_DIR"
python -u ensemble_train_predict_gpu.py
echo
# End of script
