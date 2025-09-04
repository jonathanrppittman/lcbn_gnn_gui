#!/bin/bash -l
#SBATCH --job-name GUINCandaNeuroGraphTest
#SBATCH --output /isilon/datalake/lcbn_research/final/beach/JonathanP/GUINCandaNeuroGraphTest.txt
#SBATCH --error /isilon/datalake/lcbn_research/final/beach/JonathanP/GUINCandaNeuroGraphTest.txt
#SBATCH --nodes 1
#SBATCH -p gpu-h100
#SBATCH --gpus 2
##SBATCH --mem 1850000M
#SBATCH --mem 700000M
#SBATCH --time 4:00:00

module purge
module load cuda-toolkit/11.8.0

conda activate NeuroGraph

# The command will be inserted below by the GUI application
# <<< COMMAND HERE >>>
