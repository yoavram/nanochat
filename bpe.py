import pickle


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
    ids = [encoder[c] for c in text]
    merges = []

    while len(vocab) < vocab_size:
        # Count adjacent pairs
        counts = {}
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1
        if not counts:
            break

        # Merge the most frequent pair
        best = max(counts, key=counts.get)
        new_tok = vocab[best[0]] + vocab[best[1]]
        new_id = len(vocab)
        vocab.append(new_tok)
        encoder[new_tok] = new_id
        merges.append(best)

        # Apply merge to the token stream
        merged, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        ids = merged

        if verbose and len(vocab) % 64 == 0:
            print(f"  vocab={len(vocab):4d}  tokens={len(ids):,}")

    if verbose:
        print(f"Training complete: vocab_size={len(vocab)}, "
              f"tokens={len(ids):,} (compression ratio {len(text)/len(ids):.2f}×)")
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
