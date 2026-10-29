#!/bin/bash
#SBATCH -J m2_ramer_embed
#SBATCH -p L40
#SBATCH -N 1
#SBATCH -n 7
#SBATCH --ntasks-per-node=7
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40:1
#SBATCH --exclude=gpu4015,gpu4016
#SBATCH --time=1-00:00:00
#SBATCH -o logs/protein.%j.out
#SBATCH -e logs/protein.%j.err
set -eo pipefail
source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate /share/home/u17015/.conda/envs/cpdiffusion
set -u
cd /share/home/u17015/data/GraphKcat/catapro_model2_ramer_cluster40_20261028
export OMP_NUM_THREADS=7 OPENBLAS_NUM_THREADS=7 MKL_NUM_THREADS=7 PYTHONHASHSEED=668 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -B verify_deployment.py
python -B -u extract_ramer.py
