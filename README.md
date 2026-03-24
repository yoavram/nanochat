# LLMs — Course Notebooks

Graduate-level course on large language model architecture and agents. Picks up where [DataSciPy](https://github.com/yoavram/DataSciPy) leaves off (char-level RNN/GRU/Transformer in JAX) and moves to real but tiny LLMs.

## Notebooks

The notebooks form a sequence. Run them in order.

- [`bpe-tokenizer.ipynb`](bpe-tokenizer.ipynb) — Byte Pair Encoding tokenizer from scratch: algorithm, vocabulary analysis, compression ratio
- [`nanochat.ipynb`](nanochat.ipynb) — GPT-like transformer in pure JAX: BPE tokenization, RMSNorm, RoPE, QK-norm, pretraining on TinyStories
- [`nanochat-sft.ipynb`](nanochat-sft.ipynb) — Supervised fine-tuning: response-masked loss, `[INST]…[/INST]` chat format
- [`nanochat-grpo.ipynb`](nanochat-grpo.ipynb) — GRPO reinforcement learning: group sampling, clipped surrogate, KL penalty, verifiable reward
- [`minisweagent.ipynb`](minisweagent.ipynb) — LLM agents with Ollama: ReAct loop, tool registry, mini-SWE-agent architecture


## Setup

Dependencies are managed with [pixi](https://pixi.sh) (conda + PyPI resolver).

```bash
# Install pixi (once)
curl -fsSL https://pixi.sh/install.sh | sh

# Install environment
pixi install

# Launch JupyterLab
pixi run jupyter lab
```

Key dependencies: JAX, NumPy, Matplotlib, Pandas. See [`pixi.toml`](pixi.toml) for the full list.

> The environment targets **macOS arm64** (Apple Silicon). To run on Linux or with a GPU, edit the `platforms` field in `pixi.toml` and adjust the JAX install accordingly (`jax[cuda12]` for CUDA).

## Data

Notebooks download data automatically on first run:

- **TinyStories** validation split (~10 MB) — used in `bpe-tokenizer.ipynb` and `nanochat.ipynb`
- **Ollama** with `qwen2.5:7b` — required for `minisweagent.ipynb` ([install Ollama](https://ollama.com))

## Checkpoint files

Each notebook saves a checkpoint that the next notebook loads:

```
bpe-tokenizer.ipynb  →  bpe_tokenizer.pkl
                                ↓
nanochat.ipynb       →  nanochat_checkpoint.pkl
                                ↓
nanochat-sft.ipynb   →  nanochat_sft_checkpoint.pkl
                                ↓
nanochat-grpo.ipynb  →  nanochat_grpo_checkpoint.pkl
```

## Key concepts introduced

- Byte Pair Encoding (BPE) — `bpe-tokenizer.ipynb`
- RMSNorm, RoPE, QK-norm — `nanochat.ipynb`
- Pretraining on TinyStories — `nanochat.ipynb`
- Response-masked SFT loss — `nanochat-sft.ipynb`
- GRPO, group-relative advantage — `nanochat-grpo.ipynb`
- ReAct agent loop — `minisweagent.ipynb`

## Architecture

All nanochat notebooks use **pure-functional JAX** style: no Flax or Equinox. Model parameters are plain Python dicts (JAX pytrees); `forward(params, x, cos, sin, mask)` is a pure function. See [CLAUDE.md](CLAUDE.md) for developer notes.
