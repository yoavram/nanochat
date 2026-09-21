"""Train one character model at one seed and write its bits-per-character.

    python charlm_run.py <model> <seed>

model is one of rnn, gru, transformer and their -3L variants. Everything about
the experiment -- corpus split, budget, schedule, widths -- comes from
charlm_config.json and charlm_split.npz, which are produced by RNN.ipynb, so a
cluster run and a laptop run are the same experiment by construction.

Writes results/<model>_<seed>.json. Touches nothing else.
"""
import json, os, sys, time
import jax, jax.numpy as jnp, numpy as np, optax

MODEL, SEED = sys.argv[1], int(sys.argv[2])
HERE = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(HERE, 'charlm_config.json')))
sp = np.load(os.path.join(HERE, 'charlm_split.npz'), allow_pickle=False)
train_tokens, val_tokens = sp['train'], sp['val']
V = VOCAB = len(sp['vocab'])
T, B = C['context_length'], C['batch_size']
# CHARLM_STEPS exists only so a smoke job can run a handful of steps; the
# published numbers always come from the config.
STEPS = int(os.environ.get('CHARLM_STEPS', C['steps']))

# The six published models take their widths straight from the config, so a
# cluster run and a notebook run are bit-for-bit the same setup. Any other depth
# (`gru-6L`, say) is a sweep: solve the width that holds the same budget.
PUBLISHED = {'rnn': (C['rnn_h_size'], 1), 'gru': (C['gru_h_size'], 1),
             'transformer': (C['tf_d_model'], 1),
             'rnn-3L': (C['rnn_h_size_3l'], C['n_layers_deep']),
             'gru-3L': (C['gru_h_size_3l'], C['n_layers_deep']),
             'transformer-3L': (C['tf_d_model_3l'], C['n_layers_deep'])}


def solve_width(arch, L, budget):
    """Smallest-error width holding `budget` parameters at depth L."""
    V, T_ = VOCAB, C['context_length']
    if arch == 'rnn':
        f, step = lambda h: h*V + h*h + h + (L-1)*(2*h*h + h) + V*h + V, 1
    elif arch == 'gru':
        f, step = lambda h: 3*h*V + 3*h*h + 3*h + (L-1)*(6*h*h + 3*h) + V*h + V, 1
    else:
        # d_model must stay divisible by the head count
        f, step = lambda d: V*d + T_*d + L*(12*d*d + 5*d) + d*V + V, C['tf_n_heads']
    return min(range(step, 2000, step), key=lambda x: abs(f(x) - budget))


if MODEL in PUBLISHED:
    WIDTH, NLAYERS = PUBLISHED[MODEL]
else:
    arch, _, tail = MODEL.partition('-')
    NLAYERS = int(tail.rstrip('Ll'))
    WIDTH = solve_width(arch, NLAYERS, C['param_budget'])


def layer_norm(x, eps=1e-6):
    m = jnp.mean(x, -1, keepdims=True)
    v = jnp.mean((x - m) ** 2, -1, keepdims=True)
    return (x - m) / jnp.sqrt(v + eps)


def rec_init(key, h, L, gated):
    n = 6 if gated else 2
    k = jax.random.split(key, n * L + 1)
    ls = []
    for i in range(L):
        di = V if i == 0 else h
        s = k[n * i:n * i + n]
        ls.append({'Wxz': jax.random.normal(s[0], (h, di)) * .01,
                   'Whz': jax.random.normal(s[1], (h, h)) * .01,
                   'Wxr': jax.random.normal(s[2], (h, di)) * .01,
                   'Whr': jax.random.normal(s[3], (h, h)) * .01,
                   'Wxh': jax.random.normal(s[4], (h, di)) * .01,
                   'Whh': jax.random.normal(s[5], (h, h)) * .01,
                   'bz': jnp.zeros(h), 'br': jnp.zeros(h), 'bh': jnp.zeros(h)}
                  if gated else
                  {'Wx': jax.random.normal(s[0], (h, di)) * .01,
                   'Wh': jax.random.normal(s[1], (h, h)) * .01, 'b': jnp.zeros(h)})
    return {'layers': ls, 'Why': jax.random.normal(k[-1], (V, h)) * .01,
            'by': jnp.zeros(V)}


def rec_logits(p, ids, h, gated):
    def run(L, seq, first):
        def cell(st, x):
            if gated:
                xz = L['Wxz'][:, x] if first else L['Wxz'] @ x
                xr = L['Wxr'][:, x] if first else L['Wxr'] @ x
                xh = L['Wxh'][:, x] if first else L['Wxh'] @ x
                z = jax.nn.sigmoid(xz + L['Whz'] @ st + L['bz'])
                r = jax.nn.sigmoid(xr + L['Whr'] @ st + L['br'])
                t = jnp.tanh(xh + L['Whh'] @ (r * st) + L['bh'])
                st = layer_norm((1 - z) * st + z * t)
            else:
                pre = L['Wx'][:, x] if first else L['Wx'] @ x
                st = layer_norm(jnp.tanh(pre + L['Wh'] @ st + L['b']))
            return st, st
        return jax.lax.scan(cell, jnp.zeros(h), seq)[1]

    def one(seq):
        x = seq
        for i, L in enumerate(p['layers']):
            x = run(L, x, i == 0)
        return x @ p['Why'].T + p['by']
    return jax.vmap(one)(ids)


def tf_init(key, d, L):
    ff = 4 * d
    k = jax.random.split(key, 6 * L + 3)
    blocks = [{'Wq': jax.random.normal(k[6 * i + 0], (d, d)) * .02,
               'Wk': jax.random.normal(k[6 * i + 1], (d, d)) * .02,
               'Wv': jax.random.normal(k[6 * i + 2], (d, d)) * .02,
               'Wo': jax.random.normal(k[6 * i + 3], (d, d)) * .02,
               'W1': jax.random.normal(k[6 * i + 4], (d, ff)) * .02,
               'b1': jnp.zeros(ff),
               'W2': jax.random.normal(k[6 * i + 5], (ff, d)) * .02,
               'b2': jnp.zeros(d)} for i in range(L)]
    return {'tok': jax.random.normal(k[-3], (V, d)) * .02,
            'pos': jax.random.normal(k[-2], (T, d)) * .02, 'blocks': blocks,
            'Wout': jax.random.normal(k[-1], (d, V)) * .02, 'bout': jnp.zeros(V)}


def tf_logits(p, ids, nh):
    Bn, t = ids.shape
    d = p['tok'].shape[1]
    hd = d // nh
    x = p['tok'][ids] + p['pos'][:t]
    mask = 1e10 * (1 - jnp.tril(jnp.ones((t, t))))
    for b in p['blocks']:
        xn = layer_norm(x)
        def sh(a): return a.reshape(Bn, t, nh, hd).transpose(0, 2, 1, 3)
        Q, K, Vv = sh(xn @ b['Wq']), sh(xn @ b['Wk']), sh(xn @ b['Wv'])
        lo = Q @ K.transpose(0, 1, 3, 2) / jnp.sqrt(hd) - mask
        z = (jax.nn.softmax(lo, -1) @ Vv).transpose(0, 2, 1, 3).reshape(Bn, t, d)
        x = x + z @ b['Wo']
        xn = layer_norm(x)
        x = x + (jax.nn.relu(xn @ b['W1'] + b['b1']) @ b['W2'] + b['b2'])
    return layer_norm(x) @ p['Wout'] + p['bout']


if MODEL.startswith('transformer'):
    init_fn = lambda k: tf_init(k, WIDTH, NLAYERS)
    logit_fn = lambda p, i: tf_logits(p, i, C['tf_n_heads'])
else:
    gated = MODEL.startswith('gru')
    init_fn = lambda k: rec_init(k, WIDTH, NLAYERS, gated)
    logit_fn = lambda p, i: rec_logits(p, i, WIDTH, gated)


def windows(tok, t):
    n = (len(tok) - 1) // t
    i = np.arange(n * t).reshape(n, t)
    return tok[i].astype(np.int32), tok[i + 1].astype(np.int32)


XV, YV = windows(val_tokens, T)


def full_bpc(p, b=64):
    jf = jax.jit(lambda i: jax.nn.log_softmax(logit_fn(p, i)))
    tot = n = 0
    for i in range(0, len(XV), b):
        lp = np.asarray(jf(jnp.asarray(XV[i:i + b])))
        q = np.take_along_axis(lp, YV[i:i + b][:, :, None], -1)[:, :, 0]
        tot += -q.sum()
        n += q.size
    return float(tot / n / np.log(2))


def NLL(p, x, y):
    return -jnp.take_along_axis(jax.nn.log_softmax(logit_fn(p, x)),
                                y[..., None], -1).mean()


print(f'{MODEL} seed {SEED}: width {WIDTH}, {NLAYERS} layer(s), '
      f'{STEPS:,} steps, device {jax.devices()[0].device_kind}', flush=True)

p = init_fn(jax.random.key(SEED))
n_params = int(sum(a.size for a in jax.tree_util.tree_leaves(p)))
assert abs(n_params / C['param_budget'] - 1) < 0.06, (
    f'{n_params} parameters is not within 5% of {C["param_budget"]}')

sch = optax.warmup_cosine_decay_schedule(
    init_value=C['init_lr'], peak_value=C['peak_lr'],
    warmup_steps=C['warmup_steps'], decay_steps=STEPS)
opt = optax.chain(optax.clip_by_global_norm(C['grad_clip']),
                  optax.adam(learning_rate=sch))
st = opt.init(p)
bp = jax.value_and_grad(NLL)


@jax.jit
def step_once(p, st, x, y):
    l, g = bp(p, x, y)
    u, st = opt.update(g, st, p)
    return optax.apply_updates(p, u), st, l


XS, YS = jnp.asarray(XV[:512]), jnp.asarray(YV[:512])
jval = jax.jit(lambda p: -jnp.take_along_axis(
    jax.nn.log_softmax(logit_fn(p, XS)), YS[..., None], -1).mean())

rng = np.random.default_rng(SEED)
curve = []
t0 = time.time()
for s in range(STEPS):
    if s % max(1, STEPS // 30) == 0:
        v = float(jval(p))
        curve.append((s, v))
        print(f'  step {s:6d}  val {v:.4f} nats  ({time.time()-t0:.0f}s)', flush=True)
    idx = rng.integers(0, len(train_tokens) - T - 1, size=B)
    w = train_tokens[idx[:, None] + np.arange(T + 1)[None, :]]
    p, st, _ = step_once(p, st, jnp.asarray(w[:, :-1].astype(np.int32)),
                         jnp.asarray(w[:, 1:].astype(np.int32)))
wall = time.time() - t0
curve.append((STEPS, float(jval(p))))

bpc = full_bpc(p)
rec = dict(model=MODEL, seed=SEED, bpc=bpc, n_params=n_params, seconds=wall,
           width=WIDTH, n_layers=NLAYERS, curve=curve,
           steps=STEPS, param_budget=C['param_budget'],
           val_hash=C['val_hash'], device=jax.devices()[0].device_kind)
out_dir = os.path.join(HERE, os.environ.get('CHARLM_OUT', 'results'))
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, f'{MODEL}_{SEED}.json'), 'w') as f:
    json.dump(rec, f, indent=2)
print(f'DONE {MODEL} seed {SEED}: {bpc:.4f} bpc, {n_params:,} params, '
      f'{wall/60:.1f} min', flush=True)
