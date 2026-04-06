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
- Cell IDs in nanochat notebooks follow the pattern `cell-jax-000`, `cell-jax-001`, etc.

## Architecture patterns

**nanochat / nanochat-sft / nanochat-grpo**: Pure-functional JAX — no Flax or Equinox. `init_params(key, cfg)` returns a nested Python dict (pytree). `forward(params, x, cos, sin, mask)` is a pure function. Parameters flow explicitly into every function.

**Checkpoint chain:** `bpe-tokenizer.ipynb` → `bpe_tokenizer.pkl`; `nanochat.ipynb` → `nanochat_checkpoint.pkl`; `nanochat-sft.ipynb` → `nanochat_sft_checkpoint.pkl`; `nanochat-grpo.ipynb` → `nanochat_grpo_checkpoint.pkl`.

**minisweagent**: Ollama-based agents. Tool registry pattern, ReAct (Reasoning-Acting-Observing) loop.
