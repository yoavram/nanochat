# Large Language Models Workshop

Graduate-level course on large language model architecture and agents. Picks up where [DataSciPy](https://github.com/yoavram/DataSciPy) leaves off (vision models) and moves from simple RNNs to real (but tiny) LLMs.

Start with [`index.ipynb`](index.ipynb). It is the course landing page and notebook guide.

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

On Linux, `pixi install` pulls `jax[cuda12]`, which needs an NVIDIA driver ≥ 525. Check yours with `nvidia-smi`. On macOS it installs CPU-only JAX.

This is the canonical setup procedure; [`index.ipynb`](index.ipynb) links here rather than repeating it.

## Data

Notebooks download data automatically on first run:

- **TinyStories** validation split, `TinyStoriesV2-GPT4-valid.txt` (22.5 MB) — used in `bpe-tokenizer.ipynb` and `nanochat.ipynb`
- **Poker hands** (UCI, 24 MB) — used in `sets.ipynb`
- **Ollama** with `qwen3.5:9b` — required for `minisweagent.ipynb` ([install Ollama](https://ollama.com))

Generated checkpoints are written under `checkpoints/`.

## Architecture

All nanochat notebooks use **pure-functional JAX** style: no Flax or Equinox. Model parameters are plain Python dicts (JAX pytrees); `forward(params, x, cos, sin, mask)` is a pure function. See [CLAUDE.md](CLAUDE.md) for developer notes.

## Credit
- RNN and GRU notebooks follow Andrej Karpathy's [blogpost about RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness).
- nanochat notebooks follow Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat) repo.
