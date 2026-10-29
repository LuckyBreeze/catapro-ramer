#!/bin/bash
#SBATCH -J m2_ramer_s668
#SBATCH -p L40
#SBATCH -N 1
#SBATCH -n 7
#SBATCH --ntasks-per-node=7
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40:1
#SBATCH --exclude=gpu4015,gpu4016
#SBATCH --time=1-00:00:00
#SBATCH -o logs/train.%j.out
#SBATCH -e logs/train.%j.err
set -euo pipefail
source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate /share/home/u17015/.conda/envs/graphkcat_env
cd /share/home/u17015/data/GraphKcat/catapro_model2_ramer_cluster40_20261028
export OMP_NUM_THREADS=7 OPENBLAS_NUM_THREADS=7 MKL_NUM_THREADS=7 PYTHONHASHSEED=668 PYTHONDONTWRITEBYTECODE=1
python -B verify_deployment.py
BASE=/share/home/u17015/data/GraphKcat/catapro_residue_query_model2_20261027
ARGS=(--prepared "$BASE/prepared" --protein-cache cache/ramer_protein --molt5-cache "$BASE/cache/molt5" --output run_seed668 --device cuda:0)
python -B -u train.py "${ARGS[@]}" --mode preflight
python -B check_preflight.py
python -B -u train.py "${ARGS[@]}" --mode train
python -B -u train.py "${ARGS[@]}" --mode test
