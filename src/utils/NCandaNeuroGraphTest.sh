#!/bin/bash -l

#SBATCH --job-name GUINCandaNeuroGraphTest
#SBATCH --output=/isilon/datalake/lcbn_research/final/beach/JonathanP/NeuroGraph/outputs/NCanda500_10pct_TripleNets3_GATNeuroGraphTestOutput.txt
#SBATCH --error=/isilon/datalake/lcbn_research/final/beach/JonathanP/NeuroGraph/errors/NCanda500_10pct_TripleNets3_GATNeuroGraphTestError.txt
#SBATCH --nodes 1
#SBATCH -p gpu-h100
#SBATCH --gpus 2
##SBATCH --mem 1850000M
#SBATCH --mem 480000M
#SBATCH --time 4:00:00

module purge
module load cuda-toolkit/11.8.0

conda activate NeuroGraph

cd /isilon/datalake/lcbn_research/final/beach/JonathanP/NeuroGraph

srun python main_NCanda.py \
 --data "/isilon/datalake/lcbn_research/final/beach/JonathanP/NCandaData500_cddr15a_5pct.pt" \
 --model GCNConv \
 --device 'cuda' \
 --batch_size 16 \
 --hidden 64 \
 --seed 123 \
 --threshold 0.1 \
 --num_nodes 500 \
 --epochs 500 \
 --early_stopping 10 \
 --trip_net_num '3'
