#!/bin/bash 
#SBATCH -p med2

#SBATCH --time=10:00:00
#SBATCH --nodes=1

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G

#SBATCH --account=adamgrp

#SBATCH -o dumps/slurm/job_log_%j.output
#SBATCH -e dumps/slurm/job_log_%j.error

# Computing requirements
module purge
module load conda

# Activate the environment
conda activate proc_env

######### Define Variables for the Script #########
# These values are passed as arguments to the script

# Path to your Python script
PYTHON_SCRIPT="/home/djeutsch/Projects/seaLevelRegimes/step_3/entropy_baselabels.py" 

NUM_CLUST=$1 # Number of clusters passed as an argument
BLID=$2 # Base label ID passed as an argument

echo
# Ensure the script exits on error
set -e

# Create a unique log file for each combination
LOGFILE="/home/djeutsch/Projects/seaLevelRegimes/step_3/dumps/python/ent_nclust_NC${NUM_CLUST}_BLID${BLID}.log"
echo "Running entropy base labels Python script for:"
echo " num_cluster=${NUM_CLUST}, base_label_id = ${BLID}"
python -u "$PYTHON_SCRIPT" "$NUM_CLUST" "$BLID" > "$LOGFILE"

echo
# End of script



