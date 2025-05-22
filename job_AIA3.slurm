#!/bin/bash
#SBATCH --account=punim0811
#SBATCH --partition=gpu-a100
#SBATCH --qos=feit
#SBATCH --gres=gpu:1
#SBATCH --job-name=AIA3
#SBATCH --time=0:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-user=zhenweiw1@student.unimelb.edu.au
#SBATCH --mail-type=FAIL,END
module load PyTorch/2.1.2-CUDA-11.7.0
module load torchvision/0.13.1-CUDA-11.7.0
module load scikit-learn/1.3.1
python AI_Assignment_3.py > results_AIA3.txt
