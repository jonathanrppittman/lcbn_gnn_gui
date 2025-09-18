#!/bin/bash -l

#SBATCH --job-name GNN_NCANDA_GUI
#SBATCH --output /isilon/datalake/lcbn_research/final/beach/JonathanP/GUI_NCANDA_GNN_TEST_OUT.txt
#SBATCH --error /isilon/datalake/lcbn_research/final/beach/JonathanP/GUI_NCANDA_GNN_TEST_ERROR.txt
#SBATCH --nodes 1
#SBATCH -p gpu-h100
#SBATCH --gpus 2
##SBATCH --mem 1850000M
#SBATCH --mem 480000M
#SBATCH --time 4:00:00

module purge
module load cuda-toolkit/11.8.0

conda activate /isilon/datalake/lcbn_research/final/software/LCBN/miniconda3/envs/NeuroGraph

cd /isilon/datalake/lcbn_research/final/beach/JonathanP/NeuroGraph
#COMMAND_PLACEHOLDER