#!/bin/bash
#SBATCH -J catapro_ramer668
#SBATCH -p L40
#SBATCH -N 1
#SBATCH -n 7
#SBATCH --ntasks-per-node=7
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=gpu4015,gpu4016,gpu4020
#SBATCH -o logs/train.%j.out
#SBATCH -e logs/train.%j.err
set -eo pipefail
source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate /share/home/u17015/.conda/envs/graphkcat_env
set -u
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=668 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset PYTHONPATH
cd /share/home/u17015/data/GraphKcat/catapro_baseline_ramer_cluster40_20261029
python -B -u run_pipeline.py gpu
