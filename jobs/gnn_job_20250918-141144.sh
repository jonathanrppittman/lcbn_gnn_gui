#!/bin/bash -l
#SBATCH --job-name GUINCandaNeuroGraphTest
#SBATCH --output ./logs/out.txt
#SBATCH --error ./logs/error.txt
#SBATCH --nodes 1
#SBATCH -p gpu-h100
#SBATCH --gpus 2
##SBATCH --mem 1850000M
#SBATCH --mem 700000M
#SBATCH --time 4:00:00

module purge
module load cuda-toolkit/11.8.0

conda activate /isilon/datalake/lcbn_research/final/software/LCBN/miniconda3/envs/NeuroGraph

srun python src/utils/NCandaToTorchGraphDataGUITest.py --inputs "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDA_CorrMats500ROIs_1.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDA_CorrMats500ROIs_2.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDA_CorrMats500ROIs_3.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDA_CorrMats500ROIs_4.mat" --labels "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/LabelsTotal_500.mat" --output_dir "../NeuroGraph/data/NCanda/raw" --ROIs 500