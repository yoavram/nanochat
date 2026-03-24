# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a teaching repository for a graduate-level course (DataSciPy / Computational Biology / Data Science) covering LLM architecture and agents. All content is Jupyter notebooks.

## Environment

Dependencies are managed with [pixi](https://pixi.sh). The environment targets `osx-arm64` with Python 3.14.

```bash
pixi install          # install dependencies
pixi run jupyter lab  # launch Jupyter (no task defined; run directly)
```

Key dependencies (from `pixi.toml`): JAX, NumPy, Matplotlib, Pandas.

## Notebooks

| File | Content |
|------|---------|
| `bpe-tokenizer.ipynb` | BPE tokenizer from scratch — algorithm, vocabulary analysis, compression ratio, saves `bpe_tokenizer.pkl` |
| `nanochat.ipynb` | NanoChat GPT-like transformer in **pure JAX** — BPE tokenization (512 vocab, from scratch), RMSNorm, RoPE, QK-norm, pretraining on TinyStories |
| `nanochat-sft.ipynb` | Supervised fine-tuning on top of `nanochat.ipynb` checkpoint — response-masked loss, chat template |
| `nanochat-grpo.ipynb` | GRPO reinforcement learning on top of SFT checkpoint — group sampling, clipped surrogate, KL penalty, verifiable reward |
| `microgptjax.ipynb` | Scalar autograd engine from scratch, then tiny GPT, then same computation in JAX |
| `nanochat-torch.ipynb` | NanoChat in PyTorch (kept for reference) |
| `minisweagent.ipynb` | LLM agents with Ollama — ReAct loop, tool registry, mini-swe-agent architecture |

## Architecture Patterns

**nanochat**: Pure-functional JAX style — no OOP for model layers. `init_params(key, config)` returns a plain Python dict (pytree); `forward(params, x, cos, sin, mask)` is a pure function. Parameters are dicts of arrays passed explicitly.

**nanochat-torch**: Object-oriented PyTorch style with `nn.Module` subclasses.

**microgptjax**: Pedagogical progression — implements `Value` class for scalar reverse-mode AD, then builds a tiny GPT with it, then shows the JAX equivalent.

**minisweagent**: Ollama-based local LLM agents. Uses a tool registry pattern and ReAct (Reasoning-Acting-Observing) loop.
