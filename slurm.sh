#!/bin/bash

#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=Privacymodel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=15:00:00
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
srun python3 train.py --epochs 20 --lr_gen 1e-4 --lr_crit 1e-4
srun python3 train.py --epochs 20 --lr_gen 1e-3 --lr_crit 1e-3
srun python3 train.py --epochs 20 --lr_gen 1e-5 --lr_crit 1e-5
srun python3 train.py --epochs 20 --lr_gen 1e-4 --lr_crit 4e-4
srun python3 train.py --epochs 20 --lr_gen 4e-4 --lr_crit 1e-4
srun python3 train.py --epochs 20 --lr_gen 1e-5 --lr_crit 4e-5
srun python3 train.py --epochs 20 --lr_gen 4e-5 --lr_crit 1e-5

