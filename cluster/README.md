# Running the character-model comparison on the TAU cluster

The six-model comparison in `RNN.ipynb` / `GRU.ipynb` / `text-transformer.ipynb`
is ~2 hours of serial GPU time. On the cluster the 18 runs (6 models x 3 seeds)
go out as one array job and finish in the time of the slowest single run.

The cluster run and the notebook run are the *same* experiment: both read
`checkpoints/charlm_config.json` (budget, schedule, widths) and
`checkpoints/charlm_split.npz` (the exact train/val split). Nothing about the
experiment is defined here.

## Files

| file | role |
| --- | --- |
| `charlm_run.py` | trains one `<model> <seed>` pair, writes `results/<model>_<seed>.json` |
| `cluster-env.sh` | points pixi at `/scratch300` so nothing lands in the home quota |
| `slurm-charlm-install.sh` | one-off `pixi install --locked` on a CPU node |
| `slurm-charlm-smoke.sh` | 600-step version of all six models, to check the GPU path |
| `slurm-charlm-array.sh` | the real run: array 0-17, `model = MODELS[id/3]`, `seed = SEEDS[id%3]` |

## Procedure

```bash
ssh yoavram@slurmlogin.tau.ac.il mkdir -p /scratch300/yoavram/charlm/logs
scp pixi.toml pixi.lock cluster/*.py cluster/*.sh \
    checkpoints/charlm_split.npz checkpoints/charlm_config.json \
    yoavram@slurmlogin.tau.ac.il:/scratch300/yoavram/charlm/
ssh yoavram@slurmlogin.tau.ac.il 'cd /scratch300/yoavram/charlm && sbatch slurm-charlm-install.sh'
# then smoke, then:
ssh yoavram@slurmlogin.tau.ac.il 'cd /scratch300/yoavram/charlm && sbatch slurm-charlm-array.sh'
scp 'yoavram@slurmlogin.tau.ac.il:/scratch300/yoavram/charlm/results/*.json' checkpoints/charlm/
```

`CHARLM_STEPS` and `CHARLM_OUT` exist only for the smoke job. Published numbers
always come from the config's `steps`.

Cluster facts (account, pools, pixi location) are in the repo `CLAUDE.md`.
