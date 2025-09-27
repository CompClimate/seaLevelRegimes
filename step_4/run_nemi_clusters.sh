#!/bin/bash 
#SBATCH -p high2

#SBATCH --time=03:30:00
#SBATCH --nodes=1

#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=16

#SBATCH --account=adamgrp

#SBATCH -o dumps/slurm/job_log_%j.output
#SBATCH -e dumps/slurm/job_log_%j.error

# Computing requirements
module purge
module load mamba

# Activate the environment
eval "$(mamba shell hook --shell=bash)"
mamba activate proc_env

######### Define Variables for the Script #########
# These values are passed as arguments to the script

# Path to your Python script
PYTHON_SCRIPT="/home/djeutsch/Projects/ODRI/step_4_nemi_clusters/nemi_clusters.py" 

NUM_CLUST=$1 # Number of clusters passed as an argument
REGION=$2 # Region passed as an argument
DATA_SCALER=$3 # Data scaler passed as an argument

# Loop through each combination of ensemble and minimum distance
echo

# Ensure the script exits on error
set -e

# Create a unique log file for each combination
LOGFILE="/home/djeutsch/Projects/ODRI/step_4_nemi_clusters/dumps/python/nemi_nclust_NC${NUM_CLUST}.log"

echo "Running NEMI clusters Python script for:"
echo " num_cluster=${NUM_CLUST}, region = ${DOMAIN}, data_scaler = ${SCALER}"
python -u "$PYTHON_SCRIPT" "$NUM_CLUST" "$REGION" "$DATA_SCALER" > "$LOGFILE"

echo
# End of script

