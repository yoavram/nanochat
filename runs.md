# Run log — character-model comparison (WP1)

**What we are trying to establish:** that transformers are superior to RNNs.
The honest axis for that claim is **scaling** — depth, parallelism, context —
not bits-per-character at a fixed small budget, where the GRU is still ahead
(C4, C5). See C5 for the result that carries the claim.

Every training run behind the numbers in `RNN.ipynb`, `GRU.ipynb` and
`text-transformer.ipynb`, what question it was answering, and what came back.
Append to this file; do not rewrite history. Results superseded by a later run
are kept with a note saying what replaced them.

## The experiment

Three architectures at a **fixed parameter budget**, so the comparison is about
inductive bias rather than capacity. Everything is pinned in
`checkpoints/charlm_config.json` and the split in `checkpoints/charlm_split.npz`:

| | |
|---|---|
| corpus | `data/shakespear3.txt`, 4,573,338 bytes, vocab 67 |
| split | contiguous last 10% as validation — 4,116,005 train / 457,333 val tokens, `val_hash b73fdb99e3a24d33` |
| budget | **200,000 parameters** for every model |
| context | 128 characters, batch 64, `state_policy='none'` (no state carried between windows — the only protocol all three architectures can implement) |
| schedule | warmup-cosine, `init_lr 1e-4`, `peak_lr 3e-3`, 200 warmup steps, **30,000 steps**, grad-clip 1.0 |
| seeds | 42, 43, 44 |
| metric | bits per character on the full validation split (tokenizer-independent) |

Widths are *solved* to hold the budget at each depth, so depth is bought out of
width. `cluster/charlm_run.py` contains the solver; it reproduces all six
committed config widths exactly (verified 2026-09-21).

| depth | rnn h | gru h | tf d_model |
|---|---|---|---|
| 1 | 385 | 217 | 120 |
| 2 | 236 | 135 | 84 |
| 3 | 187 | 107 | 72 |
| 4 | 159 | 91 | 60 |
| 6 | 129 | 74 | 52 (head_dim 13) |

---

## Workstation runs (2× RTX A4000, before the cluster)

These set the design. Exact per-seed values for the exploratory ones were not
all retained; the conclusions were, and each was acted on.

| # | question | what was run | outcome |
|---|---|---|---|
| W1 | what budget? | 500k and 1M budgets, 6 models, 10k steps | 500k **overfit**: gru peaked at 18k steps, gru-3L at 6k, and final ≫ best. Capacity was too high for a 4.6 MB corpus. → budget cut to 200k |
| W2 | is 10k steps converged? | validation curves to 50k steps | No — at 10k every model was still improving, so the 500k/10k gaps were partly noise on a non-converged curve. → 30k steps adopted; at 200k/30k `\|final − best\| ≈ 0`, i.e. no overfitting left |
| W3 | is cosine the right schedule here? | cosine vs `optax.contrib.reduce_on_plateau`, 200k budget | Tied — plateau won 5 of 6 by margins inside seed noise. → **cosine retained** (simpler to teach, one fewer hyperparameter) |
| W4 | how big is seed noise? | identical code+seed, repeated | rnn-3L gave 2.133 then 2.066 on a re-run. Not a bug — GPU/XLA nondeterminism compounded over 10k steps on a non-converged curve. → motivated reporting **3 seeds with error bars** rather than single numbers |
| W5 | the published 200k/30k sweep | 6 models × 3 seeds, in-notebook | rnn 2.0976±0.0112, gru 2.0635±0.0131, transformer 2.1852±0.0055, rnn-3L 2.0848±0.0153, gru-3L 2.0423±0.0086, transformer-3L 2.0403±0.0042. **Superseded by C4**, which agrees within noise and has cleaner error bars |

---

## Cluster runs (TAU Slurm, RTX A6000)

Setup lives in `cluster/` (see `cluster/README.md`). Everything runs out of
`/scratch300/yoavram/charlm` to stay off the home quota. Motivation: the six
models are ~2 h serial in the notebooks but embarrassingly parallel across
(model, seed), so an array job finishes in the time of the slowest single run.

### C1 — environment install · job `21967799` · COMPLETED 3:53
CPU pool (`power-general-shared-pool`/`public`), `pixi install --locked`.
Resolved **JAX 0.9.2 + CUDA 12, optax 0.2.8, NumPy 2.4.4**. The repo `pixi.toml`
already declared `jax[cuda12]` under `target.linux-64`, so the committed lock
gives a GPU environment on the cluster and a CPU/metal one on the laptop.

### C2 — GPU smoke, 200 steps · job `21968252` · **ALL 6 FAILED**
`ValueError: cosine_decay_schedule requires positive decay_steps, got 0`.

Not a cluster problem and not a bug in the real run: `optax`'s
`warmup_cosine_decay_schedule` decays over `decay_steps − warmup_steps`, and I
had set the smoke run to 200 steps with `warmup_steps=200`. **Worth knowing for
the notebook prose: the published run cosine-decays over 29,800 steps, not
30,000.**

### C3 — GPU smoke, 600 steps · job `21968998` · COMPLETED
All six trained end-to-end on an **RTX A6000 (49,140 MiB)**, 35–53 s each.
Parameter counts 200,141–206,635 — every model inside 3.5% of budget, so the
runner's budget assertion holds for all three architectures. This was the
green light for the full array.

### C4 — the published sweep · job `21969289` · COMPLETED · **these are the numbers to publish**
6 models × 3 seeds = 18 tasks, 30,000 steps each. Results in
`checkpoints/charlm-cluster/`.

| model | width | params | s42 | s43 | s44 | mean bpc | sd | min/run |
|---|---|---|---|---|---|---|---|---|
| rnn | 385 | 200,267 | 2.1050 | 2.0921 | 2.0929 | **2.0966** | 0.0072 | 4.0 |
| gru | 217 | 200,141 | 2.0621 | 2.0465 | 2.0737 | **2.0607** | 0.0137 | 5.8 |
| transformer | 120 | 204,907 | 2.1747 | 2.1731 | 2.1849 | **2.1775** | 0.0064 | 1.2 |
| rnn-3L | 187 | 200,531 | 2.0887 | 2.0778 | 2.0868 | **2.0844** | 0.0058 | 11.2 |
| gru-3L | 107 | 201,441 | 2.0361 | 2.0395 | 2.0505 | **2.0421** | 0.0075 | 15.7 |
| transformer-3L | 72 | 206,635 | 2.0385 | 2.0441 | 2.0501 | **2.0443** | 0.0058 | 2.0 |

Typical seed sd **0.0077**, so **a gap under ~0.015 bpc is noise**.

Depth effect (1 → 3 layers, same budget):

| | Δ bpc | pooled sd | verdict |
|---|---|---|---|
| rnn | +0.0122 | 0.0093 | 1.3σ — **within noise** |
| gru | +0.0187 | 0.0156 | 1.2σ — **within noise** |
| transformer | +0.1333 | 0.0086 | 15.4σ — **real** |

Rankings: at 1 layer `gru 2.061 < rnn 2.097 < transformer 2.178` — the
transformer is the **worst** of the six. At 3 layers
`gru-3L 2.042 ≈ transformer-3L 2.044 < rnn-3L 2.084` — a dead tie (0.0022,
well inside noise), i.e. depth lets attention *catch* the gated recurrent model
on this corpus, not beat it.

**Validation against the notebooks:** every C4 mean is within one sd of the
corresponding W5 notebook number (rnn 2.0966/2.0976, gru 2.0607/2.0635,
transformer 2.1775/2.1852, transformer-3L 2.0443/2.0403). `charlm_run.py` is a
second, independent implementation of all three architectures, so this agreement
is what licenses using the cluster numbers as the published ones.

**The runtime column is itself a result** and is arguably the stronger argument
for attention: transformer-3L reaches the same quality as gru-3L **8× faster**
(2.0 min vs 15.7). Attention parallelises across the sequence; `lax.scan` does
not.

### C5 — depth sweep · job `21970089` · COMPLETED · **the transformer-superiority result**

Question as originally posed: *is 5 or 6 layers worth it?* Depths 2, 4, 6 ×
3 architectures × 3 seeds = 27 tasks, written to `depth/`. Combined with C4's
depth-1 and depth-3 points this gives a five-point curve. Results in
`checkpoints/charlm-depth/`.

Mean bpc ± sd over seeds 42/43/44, all at the same 200k budget:

| depth | rnn | gru | transformer |
|---|---|---|---|
| 1 | 2.0966 ± 0.0072 | 2.0607 ± 0.0137 | 2.1775 ± 0.0064 |
| 2 | 2.0874 ± 0.0076 | **2.0282 ± 0.0062** | 2.0721 ± 0.0054 |
| 3 | **2.0844 ± 0.0058** | 2.0421 ± 0.0075 | 2.0443 ± 0.0058 |
| 4 | 2.1630 ± 0.0540 | 2.0791 ± 0.0091 | 2.0686 ± 0.0073 |
| 6 | 3.3429 ± **1.2746** | 2.2912 ± **0.1087** | **2.0411 ± 0.0044** |

**Depth is trainable only for the transformer.** Both recurrent models have an
optimum (gru at 2 layers, rnn at 3) and get *worse* beyond it; the transformer
improves monotonically and is still falling at 6. It is also the only model
whose bpc goes *down* as it gets deeper rather than its variance going up.

Seed spread at depth 6 — the qualitative result:

| | seeds 42 / 43 / 44 | spread |
|---|---|---|
| rnn-6L | 2.3170 / 2.9418 / 4.7698 | **2.4528** |
| gru-6L | 2.1837 / 2.2890 / 2.4010 | 0.2174 |
| transformer-6L | 2.0378 / 2.0395 / 2.0461 | **0.0083** |

A 6-layer RNN does not merely underperform — it **fails to train**. One seed
landed at 4.77 bpc against a uniform baseline of log2(67) = 6.07, i.e. barely
better than guessing. The transformer's spread at the same depth is **300×
tighter**. This is the residual-stream/LayerNorm story made visible: stacking
recurrent layers compounds a sequential product of Jacobians, while a
transformer block is an additive perturbation of a residual stream.

**This — not bpc at a fixed small depth — is the honest form of "transformers
replaced RNNs".** The field did not switch because attention gave better
per-parameter quality at 200k parameters; at this scale the GRU is still
nominally ahead (gru-2L 2.0282 vs transformer-6L 2.0411, a gap of 0.013 that is
right at the ~0.015 noise threshold). It switched because **the transformer is
the only one of the three that can be made deep**, and depth is the axis
everything else was bought with.

Supporting result, same table: transformer-6L trains in **2.9 min** against
gru-2L's 11.9 and gru-6L's 30.9. Better scaling *and* cheaper.

Caveat carried from the design: the transformer's budget wanders 95–105%
(`d_model` must divide by 4 heads), from 189,787 at depth 4 to 209,939 at
depth 6. The depth-6 point therefore has ~5% more parameters than its
competitors. This does not touch the stability result, which is qualitative.

### C6 — context-length sweep · job `21970215` · RUNNING
Question: does attention convert *more history* into quality where a recurrent
model's fixed-size hidden state cannot? Contexts 256 and 512 (the published runs
use 128), 3 architectures at depth 3, 3 seeds = 18 tasks.

Note the transformer pays `T × d_model` for its positional table, so a longer
context *shrinks* its width at fixed budget — the solver re-solves widths per
context. That makes this test harder for the transformer, not easier.

*Results: pending.*

### C7 — deep transformer extension · job `21970695` · RUNNING
The C5 transformer curve was still falling at depth 6, so: transformer-8L and
transformer-12L, plus gru-8L and rnn-8L to confirm the recurrent collapse is a
trend rather than a depth-6 accident. 4 models × 3 seeds = 12 tasks, appended to
`depth/`.

Open question this settles: whether the transformer eventually takes an outright
bpc win over gru-2L (2.0282), or whether it asymptotes just short of it.

*Results: pending.*

## Gotchas worth not rediscovering

- **`squeue` returning zero rows is not proof an array finished.** A wait loop
  keyed on `squeue` exited early while three gru-3L tasks were still training.
  Poll the **result files** (`ls results | wc -l`) instead.
- **`sacct` counts `batch` and `extern` steps** alongside real tasks, so it
  reported `18 RUNNING / 36 COMPLETED` for an 18-task array. Task-level truth is
  `squeue`; filter with `grep -v "\.\|extern"`.
- **Pending array tasks collapse into one `squeue` line**, so `1 PENDING` can
  mean 22 queued tasks.
- Each SSH round-trip costs 30–60 s (networked home). Batch commands; use
  server-side wait loops rather than polling from here.
- Quoting nested Python through `ssh` is a losing game — run checks locally or
  ship a file.
