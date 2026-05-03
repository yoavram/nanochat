from __future__ import annotations

import argparse
import glob
import math
import pickle
from functools import partial
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ModelResult:
    model: str
    checkpoint: Path
    parameter_count: int
    seq_length: int
    windows: int
    tokens: int
    average_loss: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the character-level RNN, GRU, and transformer checkpoints by "
            "trainable parameter count and average loss over many sequence windows."
        )
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing the saved checkpoint pickle files.",
    )
    parser.add_argument(
        "--text-path",
        type=Path,
        default=Path("data/shakespear3.txt"),
        help="Training text used by the notebooks.",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=50,
        help="Sequence length used by the RNN and GRU notebooks.",
    )
    parser.add_argument(
        "--transformer-heads",
        type=int,
        default=4,
        help="Number of attention heads used by text-transformer.ipynb.",
    )
    parser.add_argument(
        "--transformer-eval-seq-length",
        type=int,
        default=50,
        help=(
            "Evaluation context length for the transformer. "
            "Use this when the checkpoint stores a larger positional embedding table "
            "than the context length actually used during training."
        ),
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional cap on the number of non-overlapping windows to evaluate per model.",
    )
    parser.add_argument(
        "--transformer-batch-size",
        type=int,
        default=32,
        help="Number of transformer windows to evaluate per batch.",
    )
    return parser.parse_args()


def load_text(text_path: Path) -> tuple[str, jnp.ndarray, dict[str, int]]:
    text = text_path.read_text()
    chars = sorted(set(text))
    char_to_int = {char: index for index, char in enumerate(chars)}
    ints = jnp.array([char_to_int[char] for char in text], dtype=jnp.int32)
    return text, ints, char_to_int


def window_count(num_tokens: int, seq_length: int, max_windows: int | None) -> int:
    count = (num_tokens - 1) // seq_length
    if max_windows is not None:
        count = min(count, max_windows)
    if count <= 0:
        raise ValueError("Not enough tokens for even one evaluation window.")
    return count


def one_hot_window(ints: jnp.ndarray, start: int, seq_length: int, vocab_size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    x_ints = ints[start : start + seq_length]
    y_ints = ints[start + 1 : start + seq_length + 1]
    x = jax.nn.one_hot(x_ints, vocab_size)
    y = jax.nn.one_hot(y_ints, vocab_size)
    return x, y


def int_windows(ints: jnp.ndarray, seq_length: int, max_windows: int | None) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    num_windows = window_count(ints.shape[0], seq_length, max_windows)
    usable_tokens = num_windows * seq_length
    x_ints = ints[:usable_tokens].reshape(num_windows, seq_length)
    y_ints = ints[1 : usable_tokens + 1].reshape(num_windows, seq_length)
    return x_ints, y_ints, num_windows


def layer_norm(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)


def count_parameters(params: Any) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(leaf.size for leaf in leaves if hasattr(leaf, "size")))


def latest_checkpoint(checkpoints_dir: Path, pattern: str) -> Path:
    paths = sorted(Path(path) for path in glob.glob(str(checkpoints_dir / pattern)))
    if not paths:
        raise FileNotFoundError(f"No checkpoints matched {pattern!r} in {checkpoints_dir}")
    return paths[-1]


@jax.jit
def rnn_loss_stream(
    params: tuple[jnp.ndarray, ...],
    x_ints: jnp.ndarray,
    y_ints: jnp.ndarray,
    h0: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    Wxh, Whh, Why, bh, by = params

    def step(h: jnp.ndarray, tokens: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        x_t, y_t = tokens
        h = jnp.tanh(Wxh[:, x_t] + Whh @ h + bh)
        h = layer_norm(h)
        log_probs = jax.nn.log_softmax(Why @ h + by)
        return h, -log_probs[y_t]

    h_final, losses = jax.lax.scan(step, h0, (x_ints, y_ints))
    loss = losses.mean()
    return loss, h_final


@jax.jit
def gru_loss_stream(
    params: tuple[tuple[jnp.ndarray, ...], ...],
    x_ints: jnp.ndarray,
    y_ints: jnp.ndarray,
    h0: tuple[jnp.ndarray, ...],
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
    def step_layer(layer_params: tuple[jnp.ndarray, ...], x_t: jnp.ndarray, h_t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        Wxz, Whz, Wxr, Whr, Wxh, Whh, Why, bz, br, bh, by = layer_params
        z = jax.nn.sigmoid(Wxz[:, x_t] + Whz @ h_t + bz)
        r = jax.nn.sigmoid(Wxr[:, x_t] + Whr @ h_t + br)
        h_candidate = jnp.tanh(Wxh[:, x_t] + Whh @ (r * h_t) + bh)
        h_t = (1 - z) * h_t + z * h_candidate
        h_t = layer_norm(h_t)
        logits = Why @ h_t + by
        return logits, h_t

    def step_sequence(h_layers: tuple[jnp.ndarray, ...], tokens: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[tuple[jnp.ndarray, ...], jnp.ndarray]:
        x_t, y_t = tokens
        next_h = []
        value = x_t
        for layer_params, h_t in zip(params, h_layers):
            value, h_next = step_layer(layer_params, value, h_t)
            next_h.append(h_next)
        loss = -jax.nn.log_softmax(value)[y_t]
        return tuple(next_h), loss

    h_final, losses = jax.lax.scan(step_sequence, h0, (x_ints, y_ints))
    loss = losses.mean()
    return loss, h_final


def self_attention(
    x: jnp.ndarray,
    W_q: jnp.ndarray,
    W_k: jnp.ndarray,
    W_v: jnp.ndarray,
    W_o: jnp.ndarray,
    n_heads: int,
) -> jnp.ndarray:
    seq_len, d_model = x.shape
    d_head = d_model // n_heads

    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v

    Q = Q.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    K = K.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    V = V.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)

    w_logits = (Q @ K.transpose(0, 2, 1)) / jnp.sqrt(d_head)
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    w_logits = w_logits - 1e10 * (1 - mask)
    w = jax.nn.softmax(w_logits, axis=-1)
    z = w @ V
    z = z.transpose(1, 0, 2).reshape(seq_len, d_model)
    return z @ W_o


def transformer_block(x: jnp.ndarray, params: dict[str, jnp.ndarray], n_heads: int) -> jnp.ndarray:
    x_norm = layer_norm(x)
    x = x + self_attention(x_norm, params["W_q"], params["W_k"], params["W_v"], params["W_o"], n_heads)
    x_norm = layer_norm(x)
    h = jax.nn.relu(x_norm @ params["W1"] + params["b1"])
    x = x + h @ params["W2"] + params["b2"]
    return x


def transformer_logits_from_indices(params: dict[str, Any], indices: jnp.ndarray, n_heads: int) -> jnp.ndarray:
    seq_len = indices.shape[0]
    hidden = params["token_emb"][indices] + params["pos_emb"][:seq_len]
    for layer_params in params["layers"]:
        hidden = transformer_block(hidden, layer_params, n_heads)
    hidden = layer_norm(hidden)
    return hidden @ params["W_out"] + params["b_out"]


@partial(jax.jit, static_argnames=("n_heads",))
def transformer_loss_window(params: dict[str, Any], x_ints: jnp.ndarray, y_ints: jnp.ndarray, n_heads: int) -> jnp.ndarray:
    logits = transformer_logits_from_indices(params, x_ints, n_heads)
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.take_along_axis(log_probs, y_ints[:, None], axis=1).squeeze(axis=1).mean()


@partial(jax.jit, static_argnames=("n_heads",))
def transformer_loss_batch(params: dict[str, Any], x_ints: jnp.ndarray, y_ints: jnp.ndarray, n_heads: int) -> jnp.ndarray:
    return jax.vmap(transformer_loss_window, in_axes=(None, 0, 0, None))(params, x_ints, y_ints, n_heads)


def evaluate_rnn(checkpoint: Path, ints: jnp.ndarray, vocab_size: int, seq_length: int, max_windows: int | None) -> ModelResult:
    with checkpoint.open("rb") as file:
        params = pickle.load(file)
    x_ints, y_ints, num_windows = int_windows(ints, seq_length, max_windows)
    h_size = params[0].shape[0]
    hidden = jnp.zeros((h_size,))
    loss, hidden = rnn_loss_stream(params, x_ints.reshape(-1), y_ints.reshape(-1), hidden)

    return ModelResult(
        model="RNN",
        checkpoint=checkpoint,
        parameter_count=count_parameters(params),
        seq_length=seq_length,
        windows=num_windows,
        tokens=num_windows * seq_length,
        average_loss=float(loss),
    )


def evaluate_gru(checkpoint: Path, ints: jnp.ndarray, vocab_size: int, seq_length: int, max_windows: int | None) -> ModelResult:
    with checkpoint.open("rb") as file:
        params = pickle.load(file)
    params = tuple(tuple(layer) for layer in params)
    x_ints, y_ints, num_windows = int_windows(ints, seq_length, max_windows)
    hidden = tuple(jnp.zeros((layer[0].shape[0],)) for layer in params)
    loss, hidden = gru_loss_stream(params, x_ints.reshape(-1), y_ints.reshape(-1), hidden)

    return ModelResult(
        model="GRU",
        checkpoint=checkpoint,
        parameter_count=count_parameters(params),
        seq_length=seq_length,
        windows=num_windows,
        tokens=num_windows * seq_length,
        average_loss=float(loss),
    )


def evaluate_transformer(
    checkpoint: Path,
    ints: jnp.ndarray,
    n_heads: int,
    max_windows: int | None,
    batch_size: int,
    eval_seq_length: int | None,
) -> ModelResult:
    with checkpoint.open("rb") as file:
        params = pickle.load(file)
    checkpoint_seq_length = params["pos_emb"].shape[0]
    seq_length = checkpoint_seq_length if eval_seq_length is None else min(eval_seq_length, checkpoint_seq_length)
    x_ints, y_ints, num_windows = int_windows(ints, seq_length, max_windows)
    total_loss = 0.0
    total_windows = 0

    for start in range(0, num_windows, batch_size):
        stop = min(start + batch_size, num_windows)
        losses = transformer_loss_batch(params, x_ints[start:stop], y_ints[start:stop], n_heads)
        total_loss += float(losses.sum())
        total_windows += stop - start

    return ModelResult(
        model="Text Transformer",
        checkpoint=checkpoint,
        parameter_count=count_parameters(params),
        seq_length=seq_length,
        windows=num_windows,
        tokens=num_windows * seq_length,
        average_loss=total_loss / total_windows,
    )


def print_results(results: list[ModelResult]) -> None:
    name_width = max(len(result.model) for result in results)
    ckpt_width = max(len(result.checkpoint.name) for result in results)
    params_width = max(len(f"{result.parameter_count:,}") for result in results)

    header = (
        f"{'Model':<{name_width}}  "
        f"{'Checkpoint':<{ckpt_width}}  "
        f"{'Parameters':>{params_width}}  "
        f"{'Seq Len':>7}  "
        f"{'Windows':>8}  {'Tokens':>8}  {'Avg Loss':>10}  {'Perplexity':>10}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        perplexity = math.exp(result.average_loss)
        print(
            f"{result.model:<{name_width}}  "
            f"{result.checkpoint.name:<{ckpt_width}}  "
            f"{result.parameter_count:>{params_width},}  "
            f"{result.seq_length:>7,}  "
            f"{result.windows:>8,}  "
            f"{result.tokens:>8,}  "
            f"{result.average_loss:>10.6f}  "
            f"{perplexity:>10.3f}"
        )


def main() -> None:
    args = parse_args()
    _, ints, char_to_int = load_text(args.text_path)
    vocab_size = len(char_to_int)

    results = [
        evaluate_rnn(
            latest_checkpoint(args.checkpoints_dir, "rnn-jax-params-*.pkl"),
            ints,
            vocab_size,
            args.seq_length,
            args.max_windows,
        ),
        evaluate_gru(
            latest_checkpoint(args.checkpoints_dir, "gru-jax-params-*.pkl"),
            ints,
            vocab_size,
            args.seq_length,
            args.max_windows,
        ),
        evaluate_transformer(
            latest_checkpoint(args.checkpoints_dir, "transformer-jax-params-*.pkl"),
            ints,
            args.transformer_heads,
            args.max_windows,
            args.transformer_batch_size,
            args.transformer_eval_seq_length,
        ),
    ]
    print_results(results)


if __name__ == "__main__":
    main()