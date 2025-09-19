#!/bin/bash -l

#SBATCH --job-name GNN_TRAIN_TEST
#SBATCH --output ./out/train_out.txt
#SBATCH --error ./out/train_err.txt
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
srun python main_NCanda.py --model GATConv --data NCandaData500_cddr15a_5pct.pt --path /isilon/datalake/lcbn_research/final/beach/JonathanP/NeuroGraph/data/NCanda/raw