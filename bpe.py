import pickle
import re


def bpe_train(text, vocab_size=512, verbose=True):
    """
    Train a BPE tokenizer on text.

    Returns:
        vocab  — list of token strings; index = token id
        merges — list of (id_a, id_b) merge rules in training order
    """
    chars = sorted(set(text))
    vocab = chars[:]
    encoder = {c: i for i, c in enumerate(chars)}
    merges = []

    # Split into words and count frequencies — standard BPE optimization.
    # Iterates over unique words (~500K) instead of the full text (~2B chars).
    # Merges cannot cross word boundaries, matching GPT-2's approach.
    words = re.findall(r'\S+|\s+', text)
    word_freq = {}
    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1
    del words

    # Convert each unique word to a tuple of token ids
    word_ids = {w: tuple(encoder[c] for c in w) for w in word_freq}

    n_merges = vocab_size - len(vocab)

    while len(vocab) < vocab_size:
        # Count adjacent pairs, weighted by word frequency
        counts = {}
        for w, freq in word_freq.items():
            ids = word_ids[w]
            for a, b in zip(ids, ids[1:]):
                counts[(a, b)] = counts.get((a, b), 0) + freq
        if not counts:
            break

        # Merge the most frequent pair
        best = max(counts, key=counts.get)
        new_tok = vocab[best[0]] + vocab[best[1]]
        new_id = len(vocab)
        vocab.append(new_tok)
        encoder[new_tok] = new_id
        merges.append(best)

        # Apply merge to affected words
        for w in word_freq:
            ids = word_ids[w]
            merged, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            word_ids[w] = tuple(merged)

        if verbose and (len(vocab) - len(chars)) % 100 == 0:
            done = len(vocab) - len(chars)
            print(f"  {done}/{n_merges} merges")

    if verbose:
        total_tokens = sum(len(word_ids[w]) * f for w, f in word_freq.items())
        print(f"Training complete: vocab_size={len(vocab)}, "
              f"tokens={total_tokens:,} (compression ratio {len(text)/total_tokens:.2f}×)")
    return vocab, merges


def bpe_encode(text, vocab, merges):
    """Convert text to a list of token ids using the given vocab and merge rules."""
    encoder = {t: i for i, t in enumerate(vocab)}
    ids = [encoder.get(c, 0) for c in text]
    for a, b in merges:
        new_id = encoder[vocab[a] + vocab[b]]
        merged, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        ids = merged
    return ids


def bpe_decode(ids, vocab):
    """Convert a list of token ids back to text."""
    return ''.join(vocab[i] for i in ids if 0 <= i < len(vocab))


def bpe_save(vocab, merges, path):
    """Save vocab and merges to a pickle file."""
    with open(path, 'wb') as f:
        pickle.dump({'vocab': vocab, 'merges': merges}, f)
    print(f"Saved tokenizer to {path}")


def bpe_load(path):
    """Load vocab and merges from a pickle file."""
    with open(path, 'rb') as f:
        state = pickle.load(f)
    return state['vocab'], [tuple(m) for m in state['merges']]
