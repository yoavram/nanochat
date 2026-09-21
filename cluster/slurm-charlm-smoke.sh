#!/bin/bash
#SBATCH --job-name=charlm-smoke
#SBATCH --account=yoavhnram-users_v2
#SBATCH --partition=gpu-dudu-tzach-yoav-pool
#SBATCH --qos=owner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --array=0-5
#SBATCH --output=/scratch300/yoavram/charlm/logs/smoke-%A_%a.out
set -euo pipefail
source /scratch300/yoavram/charlm/cluster-env.sh
cd "$CHARLM"
MODELS=(rnn gru transformer rnn-3L gru-3L transformer-3L)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export CHARLM_STEPS=600 CHARLM_OUT=smoke
exec pixi run python charlm_run.py "${MODELS[$SLURM_ARRAY_TASK_ID]}" 42
