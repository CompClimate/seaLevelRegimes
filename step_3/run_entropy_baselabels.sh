#!/bin/bash 
#SBATCH -p high2

#SBATCH --time=03:00:00
#SBATCH --nodes=1

#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=16

#SBATCH --account=adamgrp

#SBATCH -o dumps/slurm/job_log_%j.output
#SBATCH -e dumps/slurm/job_log_%j.error

# Computing requirements
module purge
module load conda
module load mamba

# Activate the environment
mamba activate proc_env
# conda activate proc_env

######### Define Variables for the Script #########
# These values are passed as arguments to the script

# Path to your Python script
PYTHON_SCRIPT="/home/djeutsch/Projects/ODRI/step_3_entropy/entropy_baselabels.py" 

NUM_CLUST=$1 # Number of clusters passed as an argument
BLID=$2 # Base label ID passed as an argument
REGION=$3 # Region passed as an argument
DATA_SCALER=$4 # Data scaler passed as an argument

echo
# Ensure the script exits on error
set -e

# Create a unique log file for each combination
LOGFILE="/home/djeutsch/Projects/ODRI/step_3_entropy/dumps/python/ent_nclust_NC${NUM_CLUST}_BLID${BLID}.log"
echo "Running entropy base labels Python script for:"
echo " num_cluster=${NUM_CLUST}, base_label_id = ${BLID}, region = ${REGION}, data_scaler = ${DATA_SCALER}"
python -u "$PYTHON_SCRIPT" "$NUM_CLUST" "$BLID" "$REGION" "$DATA_SCALER" > "$LOGFILE"

echo
# End of script



