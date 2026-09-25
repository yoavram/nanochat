"""
Build the train-split tokenizer and the tokenised pretraining corpus.

This is a pipeline script, not a second implementation: every tokenizer decision
(cleaning rule, merge learning, special token, encoding) comes from `bpe.py`, which
`bpe-tokenizer.ipynb` generates. What lives here is only the memory management needed
to put 2.23 GB through a pure-Python encoder.

Why the encoding loop is not a single `bpe_encode` call: that function builds its
segment cache per call and returns one Python list of ids. On this corpus that would
be ~1.06 billion boxed ints, tens of gigabytes. So we do the same three steps it does
- collect unique segments, encode each once, assemble - but stream the assembly into a
preallocated uint16 array. `_apply_bpe_merges` and `SEGMENT_RE` are imported, so the
encoding itself is still the notebook's code.

Outputs (both gitignored):
    checkpoints/bpe_tokenizer_train.pkl   vocab + merges, trained on a sample
    checkpoints/train_tokens.npy          uint16, documents joined by <|endoftext|>

Usage:  pixi run python build_train_artifacts.py [--sample-frac 0.10]
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

from bpe import (SEGMENT_RE, END_OF_TEXT, clean_corpus, bpe_train, bpe_save,
                 bpe_load, _apply_bpe_merges)

CORPUS = 'data/TinyStoriesV2-GPT4-train.txt'
TOKENIZER_PATH = 'checkpoints/bpe_tokenizer_train.pkl'
TOKENS_PATH = 'checkpoints/train_tokens.npy'
VOCAB_SIZE = 1024
MIN_CHAR_COUNT = 100


def log(msg, t0=None):
    el = f"  [{time.time()-t0:7.1f}s]" if t0 else ""
    print(f"{time.strftime('%H:%M:%S')}{el}  {msg}", flush=True)


def load_documents():
    t = time.time()
    log(f"reading {CORPUS} …")
    with open(CORPUS, encoding='utf-8') as f:
        raw = f.read()
    docs = [s.strip() for s in raw.split(END_OF_TEXT) if s.strip()]
    del raw
    log(f"{len(docs):,} documents", t)
    return docs


def build_tokenizer(docs, sample_frac):
    """Clean the corpus, learn merges from a sample, keep the full character set."""
    t = time.time()
    log(f"cleaning (min_count={MIN_CHAR_COUNT}) …")
    docs, keep_chars = clean_corpus(docs, min_count=MIN_CHAR_COUNT)
    log(f"{len(docs):,} documents survive, {len(keep_chars)} characters kept", t)

    if os.path.exists(TOKENIZER_PATH):
        log(f"{TOKENIZER_PATH} exists — reusing it")
        vocab, merges = bpe_load(TOKENIZER_PATH)
        return docs, vocab, merges

    n = max(1, int(len(docs) * sample_frac))
    sample = '\n'.join(docs[:n])
    t = time.time()
    log(f"learning merges on {sample_frac:.0%} of documents "
        f"({n:,} docs, {len(sample):,} chars) …")
    vocab, merges = bpe_train(sample, vocab_size=VOCAB_SIZE,
                              special_tokens=(END_OF_TEXT,), chars=keep_chars,
                              verbose=True, every=200)
    log("merges learned", t)
    os.makedirs('checkpoints', exist_ok=True)
    bpe_save(vocab, merges, TOKENIZER_PATH)
    return docs, vocab, merges


def encode_corpus(docs, vocab, merges):
    """Three passes: unique segments, encode each once, assemble into uint16."""
    encoder = {tok: i for i, tok in enumerate(vocab)}
    eot_id = encoder[END_OF_TEXT]
    merge_rules = [(a, b, encoder[vocab[a] + vocab[b]]) for a, b in merges]

    # Pass 1 - unique segments, and the total length so we can preallocate.
    t = time.time()
    log("pass 1/3: collecting unique segments …")
    seen = {}
    findall = SEGMENT_RE.findall
    for i, doc in enumerate(docs):
        for seg in findall(doc):
            if seg not in seen:
                seen[seg] = None
        if (i + 1) % 250_000 == 0:
            log(f"  {i+1:,}/{len(docs):,} docs, {len(seen):,} unique segments", t)
    log(f"{len(seen):,} unique segments", t)

    # Pass 2 - encode each unique segment exactly once.
    t = time.time()
    log(f"pass 2/3: applying {len(merge_rules)} merge rules to each unique segment …")
    cache = {}
    for i, seg in enumerate(seen):
        cache[seg] = _apply_bpe_merges([encoder[c] for c in seg], merge_rules)
        if (i + 1) % 100_000 == 0:
            log(f"  {i+1:,}/{len(seen):,} segments", t)
    del seen
    log("segments encoded", t)

    # Pass 3 - assemble. One extra id per document for the separator.
    t = time.time()
    log("pass 3/3: measuring output length …")
    total = 0
    for doc in docs:
        total += sum(len(cache[s]) for s in findall(doc)) + 1
    log(f"{total:,} tokens -> {total*2/1e9:.2f} GB as uint16", t)

    t = time.time()
    log("pass 3/3: assembling …")
    out = np.empty(total, dtype=np.uint16)
    pos = 0
    for i, doc in enumerate(docs):
        for seg in findall(doc):
            ids = cache[seg]
            out[pos:pos + len(ids)] = ids
            pos += len(ids)
        out[pos] = eot_id
        pos += 1
        if (i + 1) % 250_000 == 0:
            log(f"  {i+1:,}/{len(docs):,} docs, {pos:,} tokens", t)
    assert pos == total, (pos, total)
    log("assembled", t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-frac', type=float, default=0.10,
                    help='fraction of documents used to learn merges')
    ap.add_argument('--smoke', action='store_true',
                    help='rehearse the whole pipeline on the valid split, writing '
                         'to *_smoke paths; use before committing hours to the real run')
    args = ap.parse_args()

    global CORPUS, TOKENIZER_PATH, TOKENS_PATH
    if args.smoke:
        CORPUS = 'data/TinyStoriesV2-GPT4-valid.txt'
        TOKENIZER_PATH = 'checkpoints/bpe_tokenizer_smoke.pkl'
        TOKENS_PATH = 'checkpoints/train_tokens_smoke.npy'
        log("SMOKE RUN — valid split, throwaway output paths")

    if not os.path.exists(CORPUS):
        sys.exit(f"missing {CORPUS} — run download_data.py --train")

    t0 = time.time()
    docs = load_documents()
    docs, vocab, merges = build_tokenizer(docs, args.sample_frac)

    assert len(vocab) == VOCAB_SIZE, len(vocab)
    assert vocab[-1] == END_OF_TEXT, vocab[-1]
    assert max(len(vocab) - 1, 0) < 65536, "uint16 cannot hold this vocabulary"

    tokens = encode_corpus(docs, vocab, merges)
    np.save(TOKENS_PATH, tokens)
    chars = sum(len(d) for d in docs)
    log(f"wrote {TOKENS_PATH}: {tokens.size:,} tokens, "
        f"{os.path.getsize(TOKENS_PATH)/1e9:.2f} GB, "
        f"compression {chars/tokens.size:.3f} chars/token")
    log(f"TOTAL {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
