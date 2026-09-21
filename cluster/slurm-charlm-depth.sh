#!/bin/bash
#SBATCH --job-name=charlm-depth
#SBATCH --account=yoavhnram-users_v2
#SBATCH --partition=gpu-dudu-tzach-yoav-pool
#SBATCH --qos=owner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --array=0-26
#SBATCH --output=/scratch300/yoavram/charlm/logs/depth-%A_%a.out
set -euo pipefail
source /scratch300/yoavram/charlm/cluster-env.sh
cd "$CHARLM"

# depths 2, 4 and 6 fill in the curve between the published depth-1 and depth-3 runs
MODELS=(rnn-2L gru-2L transformer-2L rnn-4L gru-4L transformer-4L rnn-6L gru-6L transformer-6L)
SEEDS=(42 43 44)
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}

export CHARLM_OUT=depth
exec pixi run python charlm_run.py "$MODEL" "$SEED"
