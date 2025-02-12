#!/bin/bash
#$ -l h_rt=1:0:0
#$ -l h_vmem=7.5G
#$ -pe smp 8
#$ -l gpu=1
#$ -l gpu_type=volta
#$ -cwd
#$ -j y
#$ -t 78

module load anaconda3
conda activate envcaha
module load cudnn/8.1.1-cuda11.2
python CNN_gral.py
