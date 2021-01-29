#!/bin/bash

#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=Privacymodel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=20:00:00
#SBATCH --mem=32000M
#SBATCH --output=output.out

module purge
module load 2019
module load 2020
module load Python/3.8.2-GCCcore-9.3.0
module load CUDA/10.1.243
module load cuDNN/7.6.5.32-CUDA-10.1.243
module load NCCL/2.5.6-CUDA-10.1.243
module load Anaconda3/2018.12

# Your job starts in the directory where you call sbatch
source activate factai
conda activate factai
# Activate your environment
# source activate ...
# Run your code
srun python3 train.py --epochs 300 --batch_size 512 --beta1 0 --beta2 0.9
srun python3 train_adversary.py --epochs 180 --batch_size 512 --checkpoint "logs/lightning_logs/version_141/checkpoints/epoch=299.ckpt" --attack_model "inversion2"
srun python3 train_adversary.py --epochs 150 --batch_size 512 --checkpoint "logs/lightning_logs/version_141/checkpoints/epoch=299.ckpt" --attack_model "inversion1"

srun python3 train.py --epochs 1200 --batch_size 512
srun python3 train_adversary.py --epochs 180 --batch_size 512 --checkpoint "logs/lightning_logs/version_142/checkpoints/epoch=1199.ckpt" --attack_model "inversion2"
srun python3 train_adversary.py --epochs 150 --batch_size 512 --checkpoint "logs/lightning_logs/version_142/checkpoints/epoch=1199.ckpt" --attack_model "inversion1"


srun python3 train.py --random_swap True --epochs 1200 --batch_size 512
srun python3 train_adversary.py --epochs 180 --batch_size 512 --random_swap True --checkpoint "logs/lightning_logs/version_143/checkpoints/epoch=1199.ckpt" --attack_model "inversion_2"
srun python3 train_adversary.py --epochs 150 --batch_size 512 --random_swap True --checkpoint "logs/lightning_logs/version_143/checkpoints/epoch=1199.ckpt" --attack_model "inversion_2"


