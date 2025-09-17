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

conda activate /isilon/datalake/lcbn_research/final/software/LCBN/miniconda3/envs/NeuroGraph

srun python src/utils/NCandaToTorchGraphDataGUITest.py --inputs "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDa_CorrMats998ROIs_1.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDa_CorrMats998ROIs_2.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDa_CorrMats998ROIs_3.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDa_CorrMats998ROIs_4.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDa_CorrMats998ROIs_6.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDa_CorrMats998ROIs_5.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDa_CorrMats998ROIs_7.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDa_CorrMats998ROIs_8.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDa_CorrMats998ROIs_9.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDa_CorrMats998ROIs_10.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDa_CorrMats998ROIs_11.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDa_CorrMats998ROIs_12.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDa_CorrMats998ROIs_13.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDa_CorrMats998ROIs_14.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDa_CorrMats998ROIs_15.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDa_CorrMats998ROIs_16.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDa_CorrMats998ROIs_17.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDa_CorrMats998ROIs_18.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDa_CorrMats998ROIs_19.mat" --labels "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/LabelsTotal_998.mat" --output_dir "." --ROIs 998