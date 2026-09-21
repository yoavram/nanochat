# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See [README.md](README.md) for the user-facing overview, setup instructions, and notebook descriptions.

Don't add co-author note to git commit messages.

Use Notebook tool to read/write notebooks.

## Environment

`pixi install` then `pixi run jupyter lab`. Platform: `osx-arm64`, Python 3.14. Key deps: JAX, NumPy, Matplotlib, Pandas.

On the workstation, run everything through `~/.pixi/bin/pixi run …`, which resolves to
`.pixi/envs/default` inside the repo (jax 0.9.2, GPU backend, 2× RTX A4000). The system
`python3` has no numpy, so nothing falls through to it silently. All notebooks declare the
`python3` kernelspec, which under `pixi run jupyter lab` is that local env.
`TF_CPP_MIN_LOG_LEVEL=3` silences XLA autotuner chatter that otherwise pollutes committed
outputs. The two GPUs are shared with other users — check `nvidia-smi` before a long run and
pin with `CUDA_VISIBLE_DEVICES` if someone else is on one.

## TAU Slurm cluster

For work too big for the workstation, jobs go to the TAU cluster. The runbook is a skill:

- **Plugin:** `cluster-agent@yoavram-lab-tools`, installed at user scope from
  `git@github.com:yoavram-lab/cluster-agent-skill.git`. Invoke it with the `cluster-agent`
  skill before doing cluster work; it covers partition discovery, submission, queue and
  accounting checks. Refresh with `claude plugin marketplace update yoavram-lab-tools`.
  A plugin installed mid-session is not in that session's skill registry — restart, or read
  its `SKILL.md` directly.
- **HTTPS clone of that repo fails** (no credential helper); add the marketplace by its
  **SSH** URL.

Facts verified on this machine (2026-09-21) — the skill says never to guess these, so
re-run `check_my_partitions` rather than trusting this list if anything looks off:

| | |
|---|---|
| SSH target | `yoavram@slurmlogin.tau.ac.il` — **the cluster user is `yoavram`**, not the workstation's `jupyter-yoavram`; using the local name gives `Permission denied` |
| Login host | `powerslurm-login.tau.ac.il`, key-based auth via `id_ed25519` |
| Account | `yoavhnram-users_v2` (exact string; do not "fix" it) |
| GPU pool | `gpu-dudu-tzach-yoav-pool` (A6000×8, `compute-0-420`), QOS `owner` — first choice; `gpu-general-pool`/`public` is the mixed-hardware fallback; `gpu-yoavram-pool` (A100×8) only if A100 is genuinely needed |
| CPU test pool | `power-general-shared-pool`, QOS `public` |
| pixi | `/scratch300/yoavram/.pixi/bin/pixi`; `~/nanochat/cluster-env.sh` exports `PIXI_HOME` and `PIXI_CACHE_DIR` to `/scratch300` — source it in job scripts |

**Each SSH round-trip costs ~30–60 s** because the home directory is networked
(`/a/home/cc/lifesci/yoavram`). Batch several commands into one `ssh` call rather than
chatting, and use a generous `timeout`; a 30 s cap will cut off a perfectly good login.

The cluster checkout at `~/nanochat` is **stale** — it predates the character-notebook work
(no `RNN.ipynb`, `GRU.ipynb` or `text-transformer.ipynb`) and `data/` is empty. Anything
sent there needs its corpus and inputs copied over first. Prefer submitting a standalone
runner script over `nbconvert --execute` on a notebook: array jobs parallelise across the
pool, whereas a notebook runs its models serially.

## Notebook conventions

- Edit `.ipynb` files with the `Read` and `NotebookEdit` tools. Do not use Bash Python to parse notebooks.
- For large notebooks (>10k tokens), use the Explore agent to read content, then `Grep` on `"id":` to find cell IDs before editing.
- Cell IDs in nanochat notebooks follow the pattern `cell-jax-000`, `cell-jax-001`, etc. **The three character notebooks (`RNN`, `GRU`, `text-transformer`) are nbformat 4.4 and have no cell ids at all** — `NotebookEdit` addresses their cells *positionally* as `cell-N`.

### NotebookEdit failure modes (all three have bitten this repo)

**1. Changing `cell_type` on a replace produces a structurally invalid cell.** The old cell's keys are kept, so a markdown cell retains `outputs` and `execution_count`, and a code cell ends up without them. `nbformat` then **fails hard** on the code case (`'outputs' is a required property`, notebook does not execute at all) and only **warns** on the markdown case — so the markdown version survives unnoticed until something else trips over it. After any `cell_type` change, verify:

```
pixi run python -c "
import json
nb=json.load(open('X.ipynb'))
print([(i,c['cell_type']) for i,c in enumerate(nb['cells'])
       if (c['cell_type']=='code' and ('outputs' not in c or 'execution_count' not in c))
       or (c['cell_type']=='markdown' and ('outputs' in c or 'execution_count' in c))])"
```

Repair by adding `"execution_count": null,` and `"outputs": [],` to the code cell and deleting both from the markdown cell.

**2. `insert` shifts every later positional index.** Chained inserts therefore hit the wrong targets — this silently *overwrote* a just-inserted markdown section with the next edit's content, destroying it, and nobody noticed because the replace "succeeded". Insert bottom-up, or re-derive indices after every insert, and re-read the structure before a chain of edits.

**3. `NotebookEdit` clears a cell's outputs when it rewrites the source**, and any `sed` on a notebook invalidates the tool's read state (forcing a re-read). When a notebook is going to be re-run anyway, clear all outputs first (`jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace`) — re-reads then cost a fraction as much, and the notebook is not left half-stripped.

**4. An executed notebook with a plot can become too large to edit at all.** `NotebookEdit` requires a `Read` in the same session, and `Read` refuses a file over 25k tokens — while `Edit` refuses `.ipynb` outright, so there is no fallback. The three character notebooks are ~13k tokens of source plus 44 KB of base64 PNG (~15k tokens) for a single loss curve, which puts them at ~28.7k once executed: editable before a run, locked after one. Measured 2026-09-21 (runs.md W6), where a caveat could not be added to `text-transformer.ipynb` after its run.

Consequences, in order of usefulness:

- **Land every prose and code edit before executing.** Execution is the last step, not an intermediate one.
- Plot `dpi` is the only lever that matters — downsampling a 30k-point curve does not shrink the PNG (43 KB at every 25th point vs 42 KB at every point), but `dpi=72` takes it to 27 KB and the notebook to ~22k tokens, under the limit.
- To edit an already-executed notebook: `jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace`, edit, re-execute. Budget the re-execution — for these notebooks that is 18 min (transformer) to 78 min (GRU) on an A4000.

**After any structural edit, validate before trusting it:** `json.load` the file, check the cell inventory, and re-read the region you changed. "The tool returned success" is not evidence the notebook is intact.

## Architecture patterns

**nanochat / nanochat-sft / nanochat-grpo / nanochat-chat**: Pure-functional JAX — no Flax or Equinox. `init_params(key, cfg)` returns a nested Python dict (pytree). `forward(params, x, cos, sin, mask)` is a pure function. Parameters flow explicitly into every function.

**nanochat-chat** is inference only: it trains nothing and produces no checkpoint. It loads the pretrained, SFT and GRPO checkpoints and compares them. Its generation loop uses a fixed-size token buffer so `jax.jit` compiles once instead of once per token — do not "simplify" it back to a growing context.

**Checkpoint chain:** `bpe-tokenizer.ipynb` → `bpe_tokenizer.pkl`; `nanochat.ipynb` → `nanochat_checkpoint.pkl`; `nanochat-sft.ipynb` → `nanochat_sft_checkpoint.pkl`; `nanochat-grpo.ipynb` → `nanochat_grpo_checkpoint.pkl`.

**minisweagent**: Ollama-based agents. Tool registry pattern, ReAct (Reasoning-Acting-Observing) loop.
