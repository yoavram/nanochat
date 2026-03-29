#!/usr/bin/env python
"""
nanochat_chat.py – interactive terminal chat with a trained nanochat model.

Loads a checkpoint produced by nanochat-grpo.ipynb (or sft / pretrained),
then runs a simple read-generate loop in the terminal.

Usage:
    python nanochat_chat.py                          # auto-detects best checkpoint
    python nanochat_chat.py --checkpoint path.pkl
    python nanochat_chat.py --temperature 0.8 --max-tokens 120

Type your message and press Enter. Type 'quit' or Ctrl-C to exit.
"""

import argparse
import math
import os
import pickle
import sys

import jax
import jax.numpy as jnp

from bpe import bpe_encode, bpe_decode


# ── Model ─────────────────────────────────────────────────────────────────────
# These functions are copied verbatim from nanochat-sft.ipynb.
# Pure-functional JAX: no classes, no global state — params flow into every call.

def rms_norm(g, x, eps=1e-6):
    return g * x / jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)

def precompute_rope(seq_len, head_dim, base=10000.):
    """Rotary positional embeddings: returns (cos, sin) tables of shape (T, head_dim)."""
    i = jnp.arange(0, head_dim, 2)
    angles = jnp.outer(jnp.arange(seq_len), 1.0 / (base ** (i / head_dim)))
    angles = jnp.concatenate([angles, angles], axis=-1)
    return jnp.cos(angles), jnp.sin(angles)

def apply_rope(x, cos, sin):
    d = x.shape[-1] // 2
    return x * cos + jnp.concatenate([-x[..., d:], x[..., :d]], axis=-1) * sin

def causal_mask(T):
    """Upper-triangular mask: future positions get -inf so softmax zeroes them out."""
    return jnp.where(jnp.tril(jnp.ones((T, T))), 0., -jnp.inf)

def attention_forward(p, x, cos, sin, mask):
    B, T, d = x.shape; hd = cos.shape[-1]; H = d // hd
    Q, K, V = x @ p['Wq'], x @ p['Wk'], x @ p['Wv']
    def sh(t): return t.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
    Q, K, V = sh(Q), sh(K), sh(V)
    c, s = cos[None, None], sin[None, None]
    Q = apply_rope(Q, c, s) / (jnp.linalg.norm(Q, axis=-1, keepdims=True) + 1e-6)
    K = apply_rope(K, c, s) / (jnp.linalg.norm(K, axis=-1, keepdims=True) + 1e-6)
    w = jax.nn.softmax(Q @ K.transpose(0, 1, 3, 2) / math.sqrt(hd) + mask[None, None], axis=-1)
    return (w @ V).transpose(0, 2, 1, 3).reshape(B, T, d) @ p['Wo'], w

def mlp_forward(p, x):
    return jax.nn.relu(x @ p['W1']) ** 2 @ p['W2']

def forward(params, ids, cos, sin, mask):
    """Full transformer forward pass. Returns logits of shape (B, T, vocab_size)."""
    x = params['tok_emb']['W'][ids]          # token embeddings: (B, T, d_model)
    for blk in params['blocks']:             # one iteration per transformer layer
        x = x + attention_forward(blk['attn'], rms_norm(blk['norm1']['g'], x), cos, sin, mask)[0]
        x = x + mlp_forward(blk['mlp'], rms_norm(blk['norm2']['g'], x))
    return rms_norm(params['norm_f']['g'], x) @ params['head']['W'].T  # (B, T, V)


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_token(key, logits, temperature=1.0, top_k=0, top_p=1.0):
    """
    Draw one token index from a logit vector.

    temperature : >1 → flatter distribution (more random);
                  <1 → sharper distribution (more greedy).
    top_k       : keep only the k highest-probability tokens before sampling.
    top_p       : nucleus sampling — keep the smallest set of tokens whose
                  cumulative probability exceeds p.
    """
    logits = logits / temperature

    # top-k: zero out everything below the k-th largest logit
    if top_k > 0:
        kth = jnp.sort(logits)[-top_k]
        logits = jnp.where(logits < kth, -jnp.inf, logits)

    probs = jax.nn.softmax(logits)

    # top-p (nucleus): zero out low-probability tail tokens
    if top_p < 1.0:
        sorted_idx = jnp.argsort(-probs)
        cumsum = jnp.cumsum(probs[sorted_idx])
        keep = jnp.concatenate([jnp.array([True]), cumsum[:-1] <= top_p])
        mask = jnp.zeros_like(probs).at[sorted_idx].set(keep.astype(probs.dtype))
        probs = probs * mask
        probs = probs / (probs.sum() + 1e-9)

    return jax.random.choice(key, probs.shape[0], p=probs)


def generate(params, vocab, merges, prompt, cfg,
             max_new_tokens=80, temperature=1.0, top_k=0, top_p=1.0, key=None):
    """
    Autoregressive token generation.

    Encodes the prompt with BPE, then repeatedly:
      1. runs the model on the current token sequence,
      2. takes the logits at the *last* position,
      3. samples one new token,
      4. appends it to the sequence.

    Returns only the newly generated text (not the prompt).
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    seq_len = cfg.get('seq_len', 128)

    ids = bpe_encode(prompt, vocab, merges)
    prompt_len = len(ids)

    for _ in range(max_new_tokens):
        ctx = ids[-seq_len:]          # keep within context window
        T = len(ctx)
        cos, sin = precompute_rope(T, cfg['head_dim'])
        logits = forward(params, jnp.array([ctx]), cos, sin, causal_mask(T))[0, -1]
        key, sub = jax.random.split(key)
        ids.append(int(sample_token(sub, logits, temperature, top_k, top_p)))

    response_ids = ids[prompt_len:]   # strip the prompt tokens
    return bpe_decode(response_ids, vocab)


# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint(path):
    """Load params, cfg, vocab, and BPE merges from a pickle checkpoint."""
    with open(path, 'rb') as f:
        s = pickle.load(f)
    vocab = s['vocab']
    merges = [tuple(m) for m in s['merges']]
    return jax.tree_util.tree_map(jnp.array, s['params']), s['cfg'], vocab, merges


# ── Chat loop ─────────────────────────────────────────────────────────────────

# Preference order: use the most-trained checkpoint that exists.
CHECKPOINTS = [
    'nanochat_grpo_checkpoint.pkl',
    'nanochat_sft_checkpoint.pkl',
    'nanochat_checkpoint.pkl',
]

def find_checkpoint():
    for path in CHECKPOINTS:
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description='Chat with a nanochat model')
    parser.add_argument('--checkpoint', default=None, help='Path to checkpoint .pkl')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--max-tokens', type=int, default=100)
    parser.add_argument('--top-k', type=int, default=0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    ckpt_path = args.checkpoint or find_checkpoint()
    if ckpt_path is None:
        sys.exit('No checkpoint found. Train a model first or pass --checkpoint.')

    print(f'Loading {ckpt_path} ...', end=' ', flush=True)
    params, cfg, vocab, merges = load_checkpoint(ckpt_path)
    print(f'done  (vocab={len(vocab)}, seq_len={cfg["seq_len"]})')
    print(f'Backend: {jax.default_backend()} | temperature={args.temperature} | max_tokens={args.max_tokens}')
    print("Type your message. 'quit' or Ctrl-C to exit.\n")

    key = jax.random.PRNGKey(args.seed)

    while True:
        try:
            user_input = input('You: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nBye!')
            break

        if not user_input or user_input.lower() in ('quit', 'exit'):
            print('Bye!')
            break

        # Wrap the user's message in the same chat template used during SFT/GRPO training.
        # The model learned to produce a response after seeing "[/INST]".
        prompt = f'[INST] {user_input} [/INST] '

        key, sub = jax.random.split(key)
        response = generate(
            params, vocab, merges, prompt, cfg,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            key=sub,
        )
        print(f'Model: {response.strip()}\n')


if __name__ == '__main__':
    main()
