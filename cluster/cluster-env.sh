# Shared cluster environment. Everything lives on /scratch300 to avoid home quota.
export CHARLM=/scratch300/yoavram/charlm
export PIXI_HOME=/scratch300/yoavram/.pixi
export PIXI_CACHE_DIR=/scratch300/yoavram/.cache/pixi
export UV_HTTP_TIMEOUT=300
export PATH="$PIXI_HOME/bin:$PATH"
