#!/bin/bash
#SBATCH --job-name=charlm
#SBATCH --account=yoavhnram-users_v2
#SBATCH --partition=gpu-dudu-tzach-yoav-pool
#SBATCH --qos=owner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --array=0-17
#SBATCH --output=/scratch300/yoavram/charlm/logs/charlm-%A_%a.out
set -euo pipefail
source /scratch300/yoavram/charlm/cluster-env.sh
cd "$CHARLM"

MODELS=(rnn gru transformer rnn-3L gru-3L transformer-3L)
SEEDS=(42 43 44)
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec pixi run python charlm_run.py "$MODEL" "$SEED"
