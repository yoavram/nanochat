#!/bin/bash
#SBATCH --job-name=charlm-install
#SBATCH --account=yoavhnram-users_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --qos=public
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch300/yoavram/charlm/logs/install-%j.out
set -euo pipefail
source /scratch300/yoavram/charlm/cluster-env.sh
mkdir -p "$PIXI_HOME" "$PIXI_CACHE_DIR"
cd "$CHARLM"
pixi install --locked
echo "--- solved environment ---"
pixi run python -c "import jax, optax, numpy; print(jax.__version__, optax.__version__, numpy.__version__)"
