# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See [README.md](README.md) for the user-facing overview, setup instructions, and notebook descriptions.

Don't add co-author note to git commit messages.

Use Notebook tool to read/write notebooks.

## Environment

`pixi install` then `pixi run jupyter lab`. Platform: `osx-arm64`, Python 3.14. Key deps: JAX, NumPy, Matplotlib, Pandas.

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

**After any structural edit, validate before trusting it:** `json.load` the file, check the cell inventory, and re-read the region you changed. "The tool returned success" is not evidence the notebook is intact.

## Architecture patterns

**nanochat / nanochat-sft / nanochat-grpo / nanochat-chat**: Pure-functional JAX — no Flax or Equinox. `init_params(key, cfg)` returns a nested Python dict (pytree). `forward(params, x, cos, sin, mask)` is a pure function. Parameters flow explicitly into every function.

**nanochat-chat** is inference only: it trains nothing and produces no checkpoint. It loads the pretrained, SFT and GRPO checkpoints and compares them. Its generation loop uses a fixed-size token buffer so `jax.jit` compiles once instead of once per token — do not "simplify" it back to a growing context.

**Checkpoint chain:** `bpe-tokenizer.ipynb` → `bpe_tokenizer.pkl`; `nanochat.ipynb` → `nanochat_checkpoint.pkl`; `nanochat-sft.ipynb` → `nanochat_sft_checkpoint.pkl`; `nanochat-grpo.ipynb` → `nanochat_grpo_checkpoint.pkl`.

**minisweagent**: Ollama-based agents. Tool registry pattern, ReAct (Reasoning-Acting-Observing) loop.
