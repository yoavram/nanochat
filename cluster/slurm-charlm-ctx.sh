#!/bin/bash
#SBATCH --job-name=charlm-ctx
#SBATCH --account=yoavhnram-users_v2
#SBATCH --partition=gpu-dudu-tzach-yoav-pool
#SBATCH --qos=owner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-17
#SBATCH --output=/scratch300/yoavram/charlm/logs/ctx-%A_%a.out
set -euo pipefail
source /scratch300/yoavram/charlm/cluster-env.sh
cd "$CHARLM"

# How much does each architecture gain from being allowed more history?
# Same 200k budget, same 3 layers; only the context window changes.
CTXS=(256 512)
MODELS=(rnn-3L gru-3L transformer-3L)
SEEDS=(42 43 44)
CTX=${CTXS[$((SLURM_ARRAY_TASK_ID / 9))]}
MODEL=${MODELS[$(( (SLURM_ARRAY_TASK_ID % 9) / 3 ))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}

export CHARLM_CONTEXT=$CTX CHARLM_OUT=ctx$CTX
exec pixi run python charlm_run.py "$MODEL" "$SEED"
