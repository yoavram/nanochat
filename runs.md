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

**Depth is trainable only for the transformer.** Both recurrent models get
*worse* past a point and the transformer does not. It is also the only model
whose bpc goes *down* as it gets deeper rather than its variance going up.

**Correction (2026-09-21, reading the table against its own noise band):** this
paragraph originally read "gru at 2 layers, rnn at 3 … the transformer improves
monotonically". Two of those three claims do not survive the 0.015 band:

- the **transformer is not monotone** — depth 3 → 4 is +0.0243, *worse*, and
  1.7× the band, so it is not dismissible. The curve falls steeply to 3, wobbles
  at 4, and reaches its best value at 6. "Falling overall and still falling at
  the end" is what the data supports.
- the **rnn has no measured optimum** — 1 → 3 is 0.0122, inside the band. It is
  flat from 1 to 3 and then collapses.
- **gru at 2 stands**: 1 → 2 is 0.0325, 4σ, and every point past 2 is worse.

Nothing downstream changes: the contrast with a collapse to 3.34 ± 1.27 is
untouched, and the depth-6 stability result is qualitative.

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

### C6 — context-length sweep · job `21970215` · COMPLETED · **attention does not convert more history into quality here**
Question: does attention convert *more history* into quality where a recurrent
model's fixed-size hidden state cannot? Contexts 256 and 512 (the published runs
use 128), 3 architectures at depth 3, 3 seeds = 18 tasks.

Note the transformer pays `T × d_model` for its positional table, so a longer
context *shrinks* its width at fixed budget — the solver re-solves widths per
context. That makes this test harder for the transformer, not easier.

Partial result (8 of 18 tasks), at context 256 against the 128 baseline:

| | 128 | 256 | change |
|---|---|---|---|
| rnn-3L | 2.0844 | 2.0684 | **better** |
| transformer-3L | 2.0443 | 2.0602 | **worse** |

The hypothesis was that attention would gain most from a longer window. So far the
opposite: the transformer pays `context_length x d_model` for its positional table,
so at a fixed budget a longer context shrinks its width (194,071 params at ctx 256,
down from 206,635). The recurrent models' widths do not depend on context, so they
get the longer window for free.

This is recorded because it is a *negative* result that the notebook now states
honestly in exercise 4 ("be prepared for the transformer to get worse") rather than
quietly dropping.

**COMPLETED, 18/18, harvested 2026-09-21.** Full result, mean ± sd over seeds
42/43/44, in `checkpoints/charlm-ctx/{256,512}/`:

| | 128 | 256 | 512 |
|---|---|---|---|
| rnn-3L | 2.0844 ± 0.0058 | **2.0684 ± 0.0264** | 2.0781 ± 0.0187 |
| gru-3L | 2.0421 ± 0.0075 | **2.0367 ± 0.0157** | 2.0505 ± 0.0018 |
| transformer-3L | **2.0443 ± 0.0058** | 2.0602 ± 0.0020 | 2.0900 ± 0.0051 |

**The hypothesis is refuted, and monotonically.** The transformer is the only
model that gets *worse* with more history, and it gets worse at every step:
+0.016 at 256, +0.046 at 512. Both recurrent models are best at 256. The
mechanism is the positional table — the solver shrinks `d_model` from 72 to 68
to 64 (206,635 → 194,071 → 189,827 parameters) to pay for it, while the
recurrent widths are independent of context and get the longer window free.

Cost scales as predicted and does not change the ranking: at 512 the transformer
takes 10.6 min against gru-3L's 61.8.

Exercise 4 in `text-transformer.ipynb` already tells students to expect this
("be prepared for the transformer to get *worse*"). It can now cite the numbers.

### C7 — deep transformer extension · job `21970695` · COMPLETED · **falsifies "still descending at 6"**
The C5 transformer curve was still falling at depth 6, so: transformer-8L and
transformer-12L, plus gru-8L and rnn-8L to confirm the recurrent collapse is a
trend rather than a depth-6 accident. 4 models × 3 seeds = 12 tasks, appended to
`depth/`. **12/12 COMPLETED, harvested 2026-09-21.**

The full curve, now seven points, rendered by `depth_table()` from the committed
files (the notebook needed no code change to pick the new points up):

| depth | rnn | gru | transformer |
|---|---|---|---|
| 1 | 2.097 ± 0.007 | 2.061 ± 0.014 | 2.178 ± 0.006 |
| 2 | 2.087 ± 0.008 | **2.028 ± 0.006** | 2.072 ± 0.005 |
| 3 | **2.084 ± 0.006** | 2.042 ± 0.008 | 2.044 ± 0.006 |
| 4 | 2.163 ± 0.054 | 2.079 ± 0.009 | 2.069 ± 0.007 |
| 6 | 3.343 ± 1.275 | 2.291 ± 0.109 | **2.041 ± 0.004** |
| 8 | 4.462 ± 0.519 | 3.369 ± 1.209 | 2.050 ± 0.015 |
| 12 | — | — | 2.056 ± 0.003 |

**The transformer has an optimum too, at 6 layers.** 8 is +0.009 (inside the
0.015 band) and 12 is +0.015 (at it), so the rise is gentle and only the 6 → 12
step clears the threshold — but the curve is no longer descending at the end,
and the answer to "does it eventually beat gru-2L (2.0282) outright" is **no**.
It bottoms out 0.013 short, which is itself inside the band: at best depth the
two are tied, exactly as C5 said.

This kills the sentence "the transformer's 2.041 ... sits on a curve still
descending" in the notebook conclusion, and the "still falling at the end" phrase
that replaced the earlier "improves monotonically". Same class of error as the
one corrected under C5: a curve read past its last measured point.

**What survives, and is strengthened:** the two recurrent models keep falling
apart while the transformer does not. rnn-8L is 4.462 — worse than the bigram
baseline (3.581) and approaching the uniform ceiling of 6.07 — and gru-8L is
3.369 with seeds spread over 2.2 bpc. Against that, the transformer moving from
2.041 to 2.056 between depth 6 and 12 is a flat line. The honest claim is
**"the transformer degrades gracefully with depth where recurrence collapses"**,
which is a better statement of the architecture's advantage than "it keeps
improving" ever was — and the cost argument is untouched: 12 transformer blocks
cost 4.7 min against 8 GRU layers' 40.3.

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

---

## Notebook rewrite (2026-09-21)

`text-transformer.ipynb` rewritten against C4 + C5. The conclusion is no longer
"transformers are better". It is a cost argument, with a hedged supporting result.
**The ordering matters and was corrected after review:**

1. **MAIN — recurrence is sequential, attention is not.** Vaswani et al. 2017
   Section 4, Table 1: self-attention is `O(1)` sequential operations against
   recurrence's `O(n)`, while being *more* arithmetic per layer (`O(n^2 d)` vs
   `O(n d^2)`). Our wall-clock column is that table measured: 6 transformer blocks
   2.9 min, 6 GRU layers 30.9 min. Attention is not cheaper in operations, it is
   cheaper in *time*, on hardware that rewards parallelism.
2. **SUPPORTING, and confounded — only the transformer converts depth into
   quality** (C5 depth table; depth-6 seed spreads rnn 2.45, gru 0.22,
   transformer 0.008).

**Why that order.** Our recurrent stacks have no residual connections while the
transformer does, so claim 2 partly measures residual streams rather than
attention, and a skip-connected deep GRU would do better than ours. Claim 1 is
untouched by that: residuals cost no parameters and no sequential steps, so a
repaired deep GRU is still 128 sequential steps per layer. **The cost argument
survives the confound; the quality argument does not.** The notebook says this in
a blockquote directly under the depth table rather than burying it in caveats.

The section states plainly that the GRU wins on bits-per-character at this scale
(gru-2L 2.028 vs transformer-6L 2.041, a gap inside the noise band) and that the
argument does not need it to lose.

Cells changed: `text-transformer` cell-40 (stacking table: 500k widths -> 200k),
cell-45 (seed band), cell-47 (the conclusion), cell-48 (exercises 1-7);
`RNN` cell-49 table + exercise 4; `GRU` cell-38 table + exercises 4 and 5.
`checkpoints/charlm_result_*.json` regenerated from C4 with a `source` field
naming the cluster job, so the in-notebook table matches the prose.

**The residual ablation was considered and deliberately not run.** The point being
taught is the cost argument, which residuals do not touch, and re-deriving a known
result (He et al. 2015) is not what this notebook is for. It is handed to the
student instead, twice: `text-transformer.ipynb` exercise 3, and `GRU.ipynb`
exercise 5 — the latter with the actual `deep_feed_forward` patch, since that is
where the code lives. Both end by pointing out that a residual GRU is still ~10x
slower at equal depth.

Also added: He et al. 2015 to the references, and a "timings are one machine"
caveat (the A6000 ratio is robust because it follows from the dependency
structure, but it would shrink on a CPU).

---

## W6 — workstation validation of C4, and the committed cell outputs · 2026-09-21

All three notebooks executed end to end on the workstation (2× RTX A4000,
`nbconvert --execute --inplace`, RNN and GRU in parallel, transformer after RNN;
57 + 78 + 18 min). **Zero cell errors** — which is itself the result being
checked, because the notebooks previously asserted a 3% budget bound that the
3-block transformer (206,635 parameters, +3.32%) could not satisfy, so the
transformer notebook could not run to completion at all. The bound is now
`BUDGET_TOL = 0.06`, the same one `cluster/charlm_run.py` uses.

A third independent execution of the experiment, on different hardware:

| model | C4 (A6000) | W6 (A4000) | Δ | C4 min | W6 min |
|---|---|---|---|---|---|
| rnn | 2.0966 ± 0.0072 | 2.0903 ± 0.0135 | −0.0063 | 4.0 | 7.5 |
| rnn-3L | 2.0844 ± 0.0058 | 2.0932 ± 0.0146 | +0.0088 | 11.2 | 15.0 |
| gru | 2.0607 ± 0.0137 | 2.0647 ± 0.0030 | +0.0040 | 5.8 | 9.9 |
| gru-3L | 2.0421 ± 0.0075 | 2.0416 ± 0.0082 | −0.0005 | 15.7 | 19.2 |
| transformer | 2.1775 ± 0.0064 | 2.1744 ± 0.0017 | −0.0032 | 1.2 | 2.3 |
| transformer-3L | 2.0443 ± 0.0058 | 2.0429 ± 0.0076 | −0.0014 | 2.0 | 4.0 |

Every gap is inside the 0.015 band, the largest being rnn-3L at 0.0088. The
orderings are unchanged, and so is every claim in the conclusion. The A4000 is
1.7–1.9× slower than the A6000 across the board, so the *ratios* the cost
argument rests on survive the hardware change — which is the empirical content
of the "timings are one machine" caveat.

**The committed `charlm_result_*.json` are still C4's**, restored with
`git checkout` after the run: `charlm_run.py` is a second implementation and C4
is the published sweep. What the notebooks now carry is their **cell outputs**,
produced by this W6 run — so the tables printed in the notebooks are W6 numbers
while the result files and the prose are C4's. They agree to within 0.009, and
the `from` column added in this pass names the machine for every row, so the
difference is visible rather than hidden.

**Reading wall-clock across that column is invalid, and the stored output shows
why:** in `text-transformer.ipynb`'s depth table, rnn depth 1 reads 7.5 min
(A4000, W6) next to depth 2's 7.0 min (A6000, C5), which looks like depth 2 being
cheaper than depth 1. Bits per character is a property of the model and survives
the machine change; minutes do not. Take cost ratios only from rows whose `from`
column agrees. `GRU.ipynb` ran last and so has a single-source main table; the
other two are mixed.

The caveat is now in the notebook prose, but getting it there took a second pass
and is worth recording. `text-transformer.ipynb` had grown to **25.5k tokens of
source with every output stripped** — past the 25k read limit on its prose
alone, with the executed PNG adding a further 15k on top. `NotebookEdit` needs a
read it could no longer get and `Edit` refuses `.ipynb`, so there was no tool
path: the edit went through the `nbformat` API instead, validated against the
cell inventory and a re-parse of every code cell, then the notebook was
re-executed (18 min, zero cell errors).

The plot cell now renders at `dpi=72`, which cut the inline PNG from 44 KB to
28 KB and the file from 129 KB to 113 KB — and did **not** help the read limit:
28,756 tokens before, 29,043 after. Bytes and tokens are not proportional across
base64, and the notebook was over the limit on its stripped prose alone, so
shrinking the figure was never going to be enough. It stays set because a
smaller file is better in git, not because it bought headroom.

`text-transformer.ipynb` therefore remains uneditable by `NotebookEdit`.
Factoring the triplicated `results_table`/`depth_table` machinery (~120 lines,
plumbing rather than pedagogy) into a module would bring the stripped source
from 25.5k to roughly 24k tokens and make the clear → edit → re-execute route
work again. Not done here — it touches all three notebooks and deserves its own
pass. **The durable rule is that prose edits land before execution**, which
costs nothing; this one cost an API-level edit and a second run.

---

## S1 — `sets.ipynb` under the metric change (WP4) · 2026-09-21 · workstation CPU

Not part of the character-model experiment; logged here because the *cross-run*
picture below exists in no single notebook. The per-class numbers themselves are
committed as outputs inside `sets.ipynb` and are not duplicated here.

Setup unchanged from the committed notebook — Yoav's "no retraining" decision:
~20k parameters each, batch 256, `optax.adamw(1e-3, wd=1e-4)`, FFN and Set
Transformer 50,000 steps, Deep Sets 200,000. Only the metric and the prose
changed. Run on CPU (`JAX_PLATFORMS=cpu`, jax 0.9.2) because the committed
outputs were CPU and the open question was a CPU-vs-GPU discrepancy.

### Deep Sets does not reproduce, and the device was not the reason

| run | device | threads | Deep Sets plain acc | macro |
|---|---|---|---|---|
| committed (original notebook) | CPU | ? | **92.1%** | not measured |
| balanced-accuracy probe, 2026-09-20 | GPU | — | 84.4% | 35.8% |
| WP4 side runner | CPU | 16 | 85.1% | 39.9% |
| **WP4 committed re-execution** | CPU | 48 | **77.6%** | **27.3%** |

The plan had attributed the 92.1 → 84.4 drop to CPU-vs-GPU float
non-determinism. That explanation does not survive: two further **CPU** runs
landed at 85.1% and 77.6%. Deep Sets sits on a long plateau and the run-to-run
variation is in *when it escapes*, not in the device.

Yoav's reading, which the evidence supports better: this looks like a **mistuned
learning rate**, not an unstable architecture. All three models share `lr=1e-3`
but escape the plateau at very different times — the Set Transformer within
~2,500 steps, the FFN around 7,500, Deep Sets not until ~30,000 and still
descending at 200,000. Deep Sets is the one that sums five per-card vectors
before `rho` sees them, so `rho`'s input scale differs from the other models'
while the init scale is a flat 0.02 everywhere. Untested — it is exercise 6.

**Consequence for the notebook:** Deep Sets' row is presented as one draw with
an explicit caveat, per Yoav's decision, not as a verdict on the architecture.

### The Set Transformer is stable, and lands on exactly 7/9

| run | macro |
|---|---|
| balanced-accuracy probe (GPU) | 77.78% |
| WP4 side runner (CPU, 16 threads) | 77.78% |
| WP4 committed re-execution (CPU, 48 threads) | 77.8% |

Three runs, two devices, identical. It scores **100% recall on all seven classes
it reaches and 0% on the two suit-defined ones** (flush, straight flush), so its
macro average is exactly 7/9 = 77.78%. Its entire error budget is flushes. The
FFN by contrast moved 72.3 → 65.6 → 68.2 across the same three runs, so *it* is
the one with a reproducibility question, not only Deep Sets.

### Flush is 0% for every model, every run

180 of 102,501 validation hands (0.18%). No model ever predicts it, including
the Set Transformer, which has the mechanism to detect it trivially — comparing
suits pairwise is the operation it already uses on ranks. Treated in the
notebook as a lesson about the objective, not the architecture.

### The stronger permutation test agrees with the weak one

The committed test applied **one** shared permutation to all 1000 hands, which
samples 1 of 120 orderings. Replacing it with an independent permutation per
hand was expected to expose more order-dependence in the FFN. It did not:
**1.10% shared against 1.00% per-hand**. The fix is still right — the old test
was weak by construction — but it must be reported as "the stronger test
agrees", not as a discovery. An exercise built on the opposite expectation was
removed rather than left to mislead.

### Gotchas

- **`sets.ipynb` crossed the 25k-token edit limit mid-package**, at 29,577
  tokens executed, and `Read` refused it — the WP-N trap, reached from under the
  limit by adding ~3k tokens of prose and one output table. Recovered by cutting
  printed training logs from up to 200 lines per model to 20 (evaluation still
  on a 100-point grid, so the loss curves are unchanged). Executed size is now
  257 KB and readable. **Watch source growth during a package, not just at the
  start of one.**
- A first background run was killed at ~10k steps with no OOM, no journal entry
  and 486 GB free. No cause was ever established. `setsid` + `nohup` made it
  moot; long runs here should be detached from the session that starts them.
- `pixi run` needs the repo as its working directory, not the scratchpad.

### S2 — why flush is 0%, and what fixes it · 2026-09-24 · workstation CPU

Follow-up to S1. The rarity explanation committed in `109d6e4` was **wrong** and is
superseded here; the notebook prose was corrected in `f753ff4`.

**The task does not require suits.** Grouping hands by rank signature alone (multiset
pattern + whether the ranks form a run) separates 7 of 10 classes perfectly, leaving two
ambiguous groups — `Nothing|Flush` and `Straight|Straight flush` — both resolvable only by
suit. So the best possible suit-blind model scores **99.82% plain / 77.78% macro (= 7/9)**,
and the Set Transformer scores **99.82% / 77.78%** with the identical unreachable set. It
found the exact optimum of the task posed. Rarity was never the cause: full house (150 val)
and four of a kind (20 val) are *rarer* than flush (180) and score 100%.

**Diagnostics on the naturally-trained models.** RMS weight per entry, both embedding tables
initialised at 0.02:

| | suit_emb | rank_emb | ratio |
|---|---|---|---|
| Set Transformer | 0.0112 (below init) | 0.1735 | 15.5× |
| Deep Sets | 0.0289 | 0.5588 | 19.3× |
| Flattened FFN | 0.2262 (above init) | 0.6553 | 2.9× |

Weight decay shrinks what no gradient defends — but **the FFN keeps a large suit channel and
still never predicts a flush**, so shrinkage is one route to suit-blindness, not the only one.
A linear probe on the *frozen* Set Transformer representation recovers **66.1% flush recall
and 83.76% macro**, above the suit-blind ceiling, at the cost of *Nothing* (34.6%) and plain
accuracy (66.97%). The information survived; the head had no reason to use it.

**Sampling regimes, 50k steps, only the minibatch sampler differs** (validation):

| | Set Transformer | Flattened FFN |
|---|---|---|
| natural | 99.82 / 77.78, flush 0% | 99.11 / 68.18, flush 0% |
| uniform over classes | **100.0 / 100.0**, flush 100% | 93.09 / 79.90, flush 100% |
| weight ∝ √count | **100.0 / 100.0**, flush 100% | **99.14 / 84.86**, flush 78.9% |

The FFN is the more informative model here and the reason the follow-up notebook uses it: the
transformer simply saturates, whereas the FFN shows a real trade (full balancing costs 6
points of plain accuracy), shows that **aggressive balancing can hurt** (four of a kind
75.0% → 60.0%), and shows the gentler √ scheme **dominating on both metrics at once**.

**Caveat that limits all of this:** straight flush has 14 training / 1 validation examples and
royal flush has 8 / 0. Uniform-over-classes puts each in ~10% of every batch — memorisation —
and validation cannot detect the failure. The Set Transformer's "100% macro" rests partly on
that and should not be quoted without it.

**Disposition.** `sets.ipynb` carries the ceiling analysis and both diagnostics and stops
there; the fix is out of scope for a transformers notebook (Yoav, 2026-09-24). Handed to
`yoavram/DataSciPy` **issue #15** with all numbers, code and caveats, for an FFN-only
imbalanced-data session on the same dataset.

---

## T1 — the BPE tokenizer, rebuilt on the valid split (WP5) · 2026-09-24 · workstation CPU

**Question:** what does the tokenizer look like when it is trained on the 22.5 MB
validation split instead of the 2.23 GB training split, and is the notebook's default
corpus change safe for the rest of the chain?

Pure Python, no accelerator. Corpus `data/TinyStoriesV2-GPT4-valid.txt`, 27,630 stories
split 27,130 train / 500 held out, 21,688,860 training characters.

| | |
|---|---|
| vocab | **1024 = 88 characters + 936 merges** |
| train time | **51 s** for 1024 merges on 21.7 M characters |
| held-out compression | **2.11×** (406,672 characters → 193,059 tokens) |
| longest token | `'unexpected'` |
| first merges | `he`, `an`, `the`, `ed`, `to`, `and`, `in`, `re`, `it`, `wa` |

Vocab-size sweep, trained on a fixed 5 MB subsample (49 s total), measured on held-out:

| vocab | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| held-out ratio | 1.323 | 1.635 | 1.892 | 2.107 |

The 5 MB subsample reproduces the full-corpus point to within 0.004, so the subsample
does not distort the curve.

### The shipped tokenizer was trained on the *train* split — WP-T's open question, answered

`checkpoints/bpe_tokenizer.pkl` as committed had **227** single-character tokens; the
valid split yields **88**. That settles WP-T's verification row: the two corpora give
different vocabularies, **0 of 1024 ids are unchanged**, and so moving the notebook's
default corpus to the valid split *does* force a retokenise and invalidates every
existing nanochat checkpoint. The old file is preserved as
`checkpoints/bpe_tokenizer_trainsplit.pkl` (gitignored) rather than lost.

### Two hazards the new raise-on-unknown encoder exposed immediately

1. **A 2 MB sweep subsample lacks `4` and `‘`, which occur in held-out text.** Under the
   old `encoder.get(c, 0)` this would have silently encoded them as token id 0 and the
   sweep would have reported slightly wrong ratios with no signal at all. The subsample
   is 5 MB for this reason, with an assertion that it covers the held-out character set.
2. **A vocabulary smaller than the corpus's character inventory is ill-posed**, not
   small: `vocab_size=64` against 81 characters returns a vocabulary of 81. The sweep
   starts at 128.

### The corpus decides which languages work, and not in the way you would guess

The 88 characters include **`é` and `ñ`** (children's-story names) but not `ß`, `ä` or
`ü`. So accented Spanish encodes cleanly at 1.27× while German is refused. A draft of
the notebook prose asserted the opposite — that accented Spanish would be refused — and
the first execution falsified it. Corrected before commit. Recorded here because the
same trap is live for anyone writing prose about tokenizer coverage: the covered set is
an accident of the corpus, so it has to be measured, not reasoned about.

Also present in the vocabulary: `\x92`, `\x93`, `\x94` — Windows-1252 smart quotes that
were mis-decoded somewhere upstream of TinyStories. Three of the 88 slots are mojibake.

### Fallback rates on the committed tokenizer

| text | result |
|---|---|
| English, in distribution | 11 tokens, 2.82×, 45% single-character |
| English, unseen words | 23 tokens, 1.61×, 52% single-character |
| Spanish, accents and all | 26 tokens, 1.27×, 73% single-character |
| German | **refused** — `ß`, `ä`, `ü` |
| English + emoji | **refused** — `🙂` |

### T2 — `<|endoftext|>` and the corpus-sampling measurement (WP5/WP-T) · 2026-09-24

**Question:** does Yoav's plan — train BPE on 10% of the train split, pretrain nanochat
on all of it — work under the raise-on-unknown encoder?

**Answer: yes, with one fix.** Full train split `data/TinyStoriesV2-GPT4-train.txt`,
2,226,845,268 characters.

| sample | distinct chars seen | chars missed | occurrences those cover |
|---|---|---|---|
| first 1% | 93 | 135 | 889 |
| first 5% | 150 | 78 | 655 |
| **first 10%** | **158** | **70** | **296** |
| first 25% | 177 | 51 | 146 |
| **full** | **228** | — | — |

296 occurrences out of 2.23 B is 1.3e-7 of the corpus — utterly negligible by
frequency, and yet each one aborts the encode of the whole corpus, because the
vocabulary is closed and `bpe_encode` now raises. **Merge statistics converge on a
sample; character inventories cannot**, since a character occurring once is either in
the sample or it is not.

Fix shipped in WP5: `bpe_train(chars=...)` overrides the character inventory, so merges
come from the sample and the character set from one full `set()` pass.

**The long tail is mostly junk.** 137 of the 228 characters occur fewer than 100 times
each; the rarest include single instances of `🎓`, `İ`, `−`, `і`, `ß`, `‌`,
``, `{`, `}`, `🤩`, `❤`, `¢`, `‚`, `ú`. Carrying all 228 spends **22% of a 1024
vocabulary** on characters and leaves ~140 fewer merges. Open question for WP-T:
clean the corpus instead.

### `<|endoftext|>` as committed (valid-split tokenizer)

| | |
|---|---|
| vocabulary | 1024 = **88 characters + 935 merges + 1 special** |
| separator id | **1023** (last), one token, round-trips exactly |
| compression cost | held-out **2.1065× → 2.1061×** (193,059 → 193,095 tokens) |

Reserved *out of* the 1024 budget, not appended to it, so nanochat's embedding and
output head keep their shape — 8.11 forces no retrain by itself.

**Two premises in `plan.md`'s own 8.11 row were wrong**, both corrected there: the vocab
does not have to grow to 1025, and `SEGMENT_RE` does **not** shred the literal —
`<|endoftext|>` contains no whitespace, so `\S+` matches all of it. The actual reason
special handling is needed is that a segment is not a token: without a reserved id the
separator is encoded character by character and merged like any word, and in this
vocabulary it raises outright, because `<`, `|` and `>` never occur in TinyStories.

### T3 — how much does cleaning the corpus buy? (WP-T) · 2026-09-24

**Question:** carrying all 228 characters of the train split spends 22% of a 1024-token
vocabulary. What does trimming the tail cost, and what does it return?

Full train split, 2,226,845,268 characters, 2,717,495 documents, 228 distinct characters.

| keep threshold | chars kept | chars dropped | occurrences affected |
|---|---|---|---|
| ≥10 | 108 | 120 | 268 (1.2e-07) |
| **≥100** | **91** | **137** | **842 (3.8e-07)** |
| ≥1,000 | 77 | 151 | 5,688 (2.6e-06) |
| ≥10,000 | 68 | 160 | 39,004 (1.8e-05) |
| ≥100,000 | 63 | 165 | 215,361 (9.7e-05) |

**The number that decides it:** at threshold 100, only **389 documents of 2,717,495
(0.0143%)** contain any rare character, totalling **356,591 characters = 0.0163% of the
corpus**. So the whole tail can be removed by dropping documents, with no text mutation
anywhere.

### Recommended policy (awaiting sign-off before the retokenise)

**keep = (count ≥ 100) ∪ (printable ASCII present in the corpus) = 104 characters**, and
drop any document containing anything else.

| | chars | merges | special |
|---|---|---|---|
| today (all 228) | 228 | 795 | 1 |
| count ≥ 100 | 91 | 932 | 1 |
| **≥100 ∪ ASCII** | **104** | **919** | **1** |

**+124 merges over today, a 15.6% larger merge budget**, for 0.0163% of the corpus. Only
552 occurrences in 2.23 B fall outside the keep set.

The ASCII clause rescues `\t # % & + = @ [ \ ] { } ~` — rare in children's stories, but
ordinary in text a student will type. Rare *in the corpus* is not rare *in the inputs*,
and under raise-on-unknown that distinction is a crash.

**Drop documents, do not edit text.** NFKD rescues only 35 of the 137 rare characters;
the other 102 have no ASCII form (CJK, `€`, `❤`, small-caps Unicode, zero-width spaces),
so any mapping scheme needs a document-dropping fallback anyway. More importantly an
in-place edit corrupts text undetectably — the same failure class as the id-0 bug 7.10
removed. Losing 0.016% visibly beats corrupting 0.016% silently.

**Correction to T1:** the `\x92 \x93 \x94` mojibake noted there is a property of the
**valid** split (3 of its 88 characters). In the train split `\x92` occurs once and
`\x93`/`\x94` never, so no special-casing is needed — the threshold absorbs them.

### T4 — cleaning moved into the notebook (WP5) · 2026-09-24

Yoav's steer: the cleanup is teaching material, so it lives in `bpe-tokenizer.ipynb`
and is emitted into `bpe.py` as `clean_corpus`, rather than sitting in a WP-T script.
`nanochat.ipynb` must clean its pretraining corpus with the *same* function — clean
differently and the corpus holds characters the vocabulary has no id for.

**Committed tokenizer (valid split, cleaned, `MIN_CHAR_COUNT=100`):**

| | |
|---|---|
| characters | 88 distinct → **79 kept** (9 dropped, 158 occurrences) |
| documents | 27,630 → **27,521** (109 dropped, 0.3945%) |
| text | 22,067,904 → 21,972,571 chars (0.4320% removed) |
| vocabulary | **79 characters + 944 merges + 1 special** = 1024 |
| dropped characters | `–‘—…é\x92\x93\x94ñ` |
| held-out compression | **2.11×** (405,879 chars → 192,470 tokens) |
| train time | 50 s |

**Separator cost, re-measured under the cleaned vocabulary:** 2.1089× with 945 merges
and no separator, **2.1088×** with 944 merges and `<|endoftext|>`. Supersedes the
2.1065×/2.1061× pair in T2, which was measured before cleaning.

### The teaching contrast: cleaning pays more on bigger corpora

| | valid | train |
|---|---|---|
| size | 22.5 MB | 2.23 GB |
| documents | 27,630 | 2,717,495 |
| distinct characters | 88 | 228 |
| kept | 79 | 104 |
| documents dropped | 109 (**0.39%**) | 389 (**0.014%**) |
| merges freed | **+9** | **+124** |

100× the text gives 2.6× the characters and almost all of the excess is junk, so the
benefit of cleaning **rises** with corpus size while its cost in documents **falls**.
That asymmetry, not the absolute numbers, is the point the notebook makes.

### Cleaning changed the out-of-vocabulary lesson, again

`é` (4 occurrences) and `ñ` (1) fall below the threshold, so the cleaned tokenizer
**refuses accented Spanish**, which the uncleaned one encoded (T1). Prose rebuilt
around it: cleaning made the tokenizer better at English and narrower everywhere else,
replacing an *accidental* boundary with a *stated* one. Both arbitrary; only one can be
written down, predicted, and changed by editing `MIN_CHAR_COUNT`.

Surviving typographic characters are worth noting: `’ “ ”` kept, `‘ – —` dropped — the
right single quote doubles as the apostrophe in `don’t` and is everywhere, while the
left single quote only opens quotations, which these stories rarely use.

**Third prose-ahead-of-measurement correction in this package.** `bpe_encode`'s error
message also said the characters "never occurred in the training corpus", false for
anything cleaning removed; it now names both causes.
