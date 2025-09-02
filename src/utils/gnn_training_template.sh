#!/bin/bash -l
#SBATCH --job-name GNNTraining
#SBATCH --output gnn_training_output.txt
#SBATCH --error gnn_training_error.txt
#SBATCH --nodes 1
#SBATCH -p gpu-h100
#SBATCH --gpus 1
#SBATCH --mem 100G
#SBATCH --time 1:00:00

# Placeholder for environment setup, if needed
# e.g., module load cuda-toolkit/11.8.0
# e.g., conda activate my-env

# The command will be inserted below by the GUI application
# <<< COMMAND HERE >>>
