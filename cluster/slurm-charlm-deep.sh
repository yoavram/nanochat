#!/bin/bash
#SBATCH --job-name=charlm-deep
#SBATCH --account=yoavhnram-users_v2
#SBATCH --partition=gpu-dudu-tzach-yoav-pool
#SBATCH --qos=owner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-11
#SBATCH --output=/scratch300/yoavram/charlm/logs/deep-%A_%a.out
set -euo pipefail
source /scratch300/yoavram/charlm/cluster-env.sh
cd "$CHARLM"

# The transformer curve was still falling at 6 layers while both recurrent
# models had already turned over and destabilised. Push the transformer further,
# and confirm the recurrent collapse is a trend and not a depth-6 accident.
MODELS=(transformer-8L transformer-12L gru-8L rnn-8L)
SEEDS=(42 43 44)
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}

export CHARLM_OUT=depth
exec pixi run python charlm_run.py "$MODEL" "$SEED"
