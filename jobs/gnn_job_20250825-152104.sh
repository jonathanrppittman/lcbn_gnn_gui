#!/usr/bin/env bash
#SBATCH --job-name=gnn_job
#SBATCH --output=./logs/%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00


python /isilon/datalake/lcbn_research/final/beach/JonathanP/lcbn_gnn_gui/src/utils/NCandaToTorchGraphDataGUITest.py --inputs "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDA_CorrMats500ROIs_1.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDA_CorrMats500ROIs_2.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDA_CorrMats500ROIs_3.mat" "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDA_CorrMats500ROIs_4.mat" --labels "/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/LabelsTotal_500.mat" --output "."
