#!/bin/bash -l
#SBATCH --job-name MakeTorchGraphData
#SBATCH --output=/isilon/datalake/lcbn_research/final/beach/JonathanP/MakeTorchGraphDataOutput.txt
#SBATCH --error=/isilon/datalake/lcbn_research/final/beach/JonathanP/MakeTorchGraphDataError.txt
#SBATCH --nodes 1
#SBATCH -p gpu-h100
#SBATCH --gpus 2
#SBATCH --mem 700000M
#SBATCH --time 4:00:00

module purge
module load cuda-toolkit/11.8.0

conda activate NeuroGraph

cd /isilon/datalake/lcbn_research/final/beach/JonathanP
##srun python MakeTorchGraphData.py
srun python NCandaToTorchGraphDataGUITest.py --threshold .1 --net_number 3 --inputs "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDA_CorrMats500ROIs_1.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDA_CorrMats500ROIs_2.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDA_CorrMats500ROIs_3.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDA_CorrMats500ROIs_4.mat" --labels "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/LabelsTotal_500.mat" --output_dir .
