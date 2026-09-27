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
| W5 | the published 200k/30k sweep | 6 models × 3 seeds, in-notebook | rnn 2.0976±0.0112, gru 2.0635±0.0131, transformer 2.1852±0.0055, rnn-3L 2.0848±0.0153, gru-3L 2.0423±0.0087, transformer-3L 2.0403±0.0042. **Superseded by C4**, which agrees within noise and has cleaner error bars |

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
| transformer | +0.1333 | 0.0087 | 15.4σ — **real** |

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

> ⚠ **Partly superseded.** The character census here is of the *raw file* (228) rather
> than document text (227), the sampling probe used a character-count prefix rather than a
> document sample (see **T5** for the shipped measurement), and the claim that `<`, `|`
> and `>` "never occur in TinyStories" holds only for the **validation** split — all three
> occur in the training split and are in its 104-character kept set.

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

> ⚠ **Superseded 2026-09-25. Every number in this section is pre-correction.** It was
> computed on the *raw file* (228 distinct characters, including the `<|endoftext|>`
> literals) and for the **frequency-only** keep rule, not the frequency ∪ printable-ASCII
> rule that shipped. Corrected values: 227 distinct characters in document text, 104 kept,
> **230 documents dropped (0.0085%)**, **+123 merges**, and the frequency-only keep is
> **88** characters, not the 91 below. See **T5**. Kept for the shape of the argument only.

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
| held-out compression | **2.11×** (405,879 chars → 192,470 tokens) — *superseded: this is the full-corpus-merge figure; since 2026-09-25 the notebook learns merges from a 10% sample and ships 192,624 tokens / 2.1071×* |
| train time | 50 s |

**Separator cost, re-measured under the cleaned vocabulary:** 2.1089× with 945 merges
and no separator, **2.1088×** with 944 merges and `<|endoftext|>`. Supersedes the
2.1065×/2.1061× pair in T2, which was measured before cleaning. **Itself superseded
2026-09-25** by the sampling change: the shipped pair is 192,621 / 192,624 tokens,
both 2.1071×.

### The teaching contrast: cleaning pays more on bigger corpora

| | valid | train |
|---|---|---|
| size | 22.5 MB | 2.23 GB |
| documents | 27,630 | 2,717,495 |
| distinct characters | 88 | **227** |
| kept | 79 | 104 |
| documents dropped | 109 (**0.39%**) | **230 (0.0085%)** |
| merges freed | **+9** | **+123** |

*(train column corrected 2026-09-25 from 228 / 389 (0.014%) / +124 — see T5.)*

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

### T5 — the shipped train-split artifacts (WP-T) · 2026-09-25 · workstation CPU

`build_train_artifacts.py`, **15.9 min total** against the 1–3 h budgeted.

| stage | time |
|---|---|
| read + split 2.23 GB into documents | 14 s |
| `clean_corpus` over 2,717,495 documents | 83 s |
| learn 919 merges on 10% of documents (218.6 M chars) | 159 s |
| collect 238,566 unique segments | 109 s |
| apply 919 merge rules to each unique segment | 95 s |
| measure + assemble 1.04 B tokens | 496 s |

**Artifacts** (both gitignored; to be shipped in a GitHub release per B6):

| | |
|---|---|
| `checkpoints/bpe_tokenizer_train.pkl` | **104 characters + 919 merges + 1 special** = 1024 |
| `checkpoints/train_tokens.npy` | **1,039,345,143 tokens**, `uint16`, **2.08 GB** |
| compression | **2.102 characters/token** |
| cleaning | 227 distinct → 104 kept (123 dropped, 546 occurrences); **230 of 2,717,495 documents dropped (0.0085%)**, 0.0098% of text |

Verified: separator count **exactly** 2,717,265 = the surviving document count; max id
1023; decodes cleanly across document boundaries. The `uint16` decision delivered 4×:
**2.08 GB against the old `int64` file's 8.5 GB**.

### Three published numbers were wrong, and the run caught them

| claimed | actual | why |
|---|---|---|
| 389 documents dropped (0.014%) | **230 (0.0085%)** | 389 was measured for the frequency-only rule; the shipped policy is frequency **∪ printable ASCII**, which rescues 16 more characters (`\t # % & + < = > @ [ \ ] { | } ~`) and so drops fewer documents |
| 228 distinct characters | **227** | 228 counted the raw file, which contains the `<|endoftext|>` literals |
| +124 merges | **+123** | follows from 227 |

All three were in the notebook's teaching table. Corrected there and in `plan.md` on the day; **the copies in T2, T3 and T4 were missed and were only corrected on 2026-09-25**, after a review pointed out that this very sentence was false about the repo it describes.

### The sampling trap, measured on the shipped pipeline

Supersedes T2's raw-file probe (which used a character-count prefix of the uncleaned
file). On the **cleaned** corpus, sampling by document:

| sample | kept characters seen | missed |
|---|---|---|
| 1% | 87/104 | 17 |
| 5% | 89/104 | 15 |
| **10%** (shipped) | **95/104** | **9** — `% < = > @ { \| } ~` |
| 25% | 101/104 | 3 — `{ \| }` |

**Every missed character is printable ASCII.** So the sampling trap and the
always-keep-ASCII clause are the same story: what a sample misses is not a random
subset but the rare tail, and for a children's-story corpus the rare tail is precisely
the punctuation a *person* would later type. Without `chars=keep_chars` the shipped
tokenizer would refuse `50% off`, `x <= y`, an email address and a dict literal — and
only at the moment someone tried. This is now the notebook's sampling section.

---

# Run log — nanochat pretraining (WP6)

## N1 — QK-norm A/B · 2026-09-26 · **decided the retrain**

**Question.** `nanochat.ipynb` normalised Q and K to unit L2 length before the
`1/sqrt(head_dim)` scale. Review finding 8.1 argued this flattens attention by
construction. The arithmetic is not in doubt — with `head_dim=64` every attention logit
lands in [-0.125, 0.125], so over a 256-token context no weight can sit more than ±13%
from 1/256 — but arithmetic does not say what it *costs*. This run measures the cost
before paying for a multi-hour retrain.

**Design.** Two runs, identical in every respect but the normalisation: same init key,
same data order, same validation batches (a fixed key, so both variants are scored on the
same windows), same optimiser and schedule. Full nanochat config — 26,223,104 params,
d_model 512, 8 heads, 8 layers, context 256, batch 32, `adamw(3e-4, wd=0.1)` — on the
train-split corpus. 4,000 steps each, run concurrently on the two A4000s.
Script: `qknorm_ab.py` (scratch; not committed — the notebook is the artifact).

| variant | val @ 4,000 | attention entropy ratio | wall-clock |
|---|---|---|---|
| **RMS** (`x / sqrt(mean(x²))`, the fix) | **1.0707** | **0.668** | 513 s |
| L2 (`x / ‖x‖₂`, the old code) | 1.5027 | **0.9995** | 555 s |

**The gap is 0.432 nats/token** — two orders of magnitude beyond anything seed noise
could produce, and visible from step 200 onwards. Val loss by step:

| step | 100 | 600 | 1100 | 1600 | 2100 | 2600 | 3100 | 3600 | 4000 |
|---|---|---|---|---|---|---|---|---|---|
| RMS | 3.292 | 1.625 | 1.353 | 1.241 | 1.180 | 1.140 | 1.110 | 1.086 | **1.071** |
| L2 | 3.438 | 2.510 | 2.001 | 1.827 | 1.711 | 1.635 | 1.576 | 1.535 | **1.503** |

**The entropy ratio is the finding, not the loss.** Mean row-entropy of the attention
distributions divided by the entropy of the uniform distribution over the positions each
row may see; 1.0 means the head averages its whole context, 0 means it has picked out one
position. Under L2 it is **0.9995**, and per-layer it reads
`[1.000, 0.999, 1.000, 0.999, 0.999, 0.999, 1.000, 1.000]` — attention is uniform to
within 0.05%, in every layer, exactly as the arithmetic predicts and regardless of what
the weights do. Under RMS it is 0.668, ranging 0.601–0.720 across layers. This is the
measurement 8.1 asked for, obtained from a live A/B rather than from the old checkpoint.

**Context for the old published number.** The original run reached val **1.069** after
23,500 steps with early stopping. The RMS variant reaches **1.071 in 4,000 steps** —
about 6× fewer steps for the same loss. So the old model was not broken, as the review's
wording ("flattens attention") might suggest: it was a "current token + almost unweighted
bag of context" model, which on TinyStories still learns a great deal. It was just paying
roughly 6× the compute for it. That tempering is now in the notebook prose.

**Decision: retrain with parameter-free RMS QK-norm.** Confirms E6 and closes 8.1.

## N2 — nanochat retrain · 2026-09-26 · **val 0.8126, against the old model's 1.069**

Budget set from the scaling law rather than a round number:
20 tokens/parameter × 26,223,104 = 524M tokens = **64,000 steps** × 32 × 256
(0.51 epochs of the 1.04B-token corpus). Fixed budget, **no early stopping** (B5);
best-checkpoint saving retained, since that is the checkpoint that ships.
One `nbconvert --execute` of the whole notebook, so the committed outputs are a single
run (8.7) rather than the composite the review found.

**This is the second launch.** The first was killed at ~20 min when the code review (N3)
landed — two of its fixes change the training trajectory, and `nbconvert --inplace`
overwrites the notebook at the end, so there was no way to fix and keep the run.

| | |
|---|---|
| wall-clock | **126.8 min** on one A4000 (0.119 s/step, slightly better than the A/B's 0.128) |
| best val | **0.8126 nats/token at step 63,000** |
| final val | 0.817 at step 64,000 — still improving, gently |
| held-out loss | 0.8087 nats/token, perplexity 2.2 |
| compression | 2.104 chars/token (separators excluded from the character count) |
| **bits per character** | **0.555** |
| attention entropy ratio | **0.687** (0.505 at layer 0 rising to 0.773 at layer 7) |

**Against the baselines** — the point of A4/8.9, and the reason the table exists:

| model | bpc |
|---|---|
| uniform over 1024 tokens | 4.754 |
| character unigram | 4.446 |
| character bigram | 3.292 |
| **nanochat, 26.2M params** | **0.555** |

**Against the old model: 1.069 → 0.8126 nats/token**, a 0.256-nat improvement, and the old
figure came from an *early-stopped* 23,500-step run while this one is a fixed 64,000-step
budget — so the comparison is not like-for-like on compute, only on outcome. The honest
statement is the one the A/B (N1) supports: same architecture and data, the normalisation
is the only difference, and it is worth ~0.43 nats at 4,000 steps.

**The entropy ratio rises monotonically with depth** (0.505 → 0.773): the early layers
attend sharply, the late layers more broadly. Under unit-L2 all eight sat at 0.999. Nothing
in the notebook predicted the *direction* of that gradient, and it is not claimed as a
result — it is simply what this checkpoint does, and it is now a figure students can look
at.

**Samples are coherent.** "Once upon a time there was a little girl named Maria. She was
only three years old and loved to explore. One day, she was walking through the park when
she came across a big pile of hay." — grammatical, narratively structured, and stopping
only because `max_new_tokens=80` ran out.

### Gotchas from this package

- **A 200-step rehearsal of the whole notebook is worth its 10 minutes.** It caught
  nothing in the end, but it is the only way to learn that the `inspect.getsource`
  module-emission cell works under `nbconvert` *before* spending 2.3 h to find out.
  `inspect.getsource` does work on a `@jax.jit`-wrapped function, and returns the
  decorator line with it, so the generated module keeps its `jit`.
- **The rehearsal overwrote `checkpoints/nanochat_{checkpoint,best}.pkl`** with its
  200-step model, destroying the old L2-normed checkpoints. No loss: WP6 invalidates them
  by design, and N1 measured the L2 variant directly rather than from that checkpoint.
  But it is a reminder that a rehearsal writes to the same paths as the real run.
- **Do not materialise the training windows.** The old code built a `(4.06M, 257)` array
  and pushed it to the device — 4 GB before a single step. Keeping the corpus flat and
  `uint16` on the host and cutting 33 KB batches out of it costs nothing measurable and
  removes the memory ceiling entirely (A3).

### N3 — adversarial code review of WP6 · 2026-09-26 · **7 bugs, one of which moved a published number**

Per the ground rule, reviewed before closing the package — and, this time, *during* the
retrain rather than after it, which is what made the fixes free. The first retrain was
**killed at ~20 min and restarted** once the review landed, because two of the fixes change
the training trajectory and `nbconvert --inplace` would have overwritten any edit anyway.

**Confirmed correct** (the half of a review that is worth as much as the bug list):
`qk_norm` leaves `‖x‖₂ = 7.9999995 = sqrt(64)`; the RoPE prose was verified
element-by-element against the code for `d=8, T=5` — **the split-half claim in 8.2 is exactly
what `apply_rope` computes** — and relative-position invariance holds to float precision;
`param_exact` is exact for three configs including a non-default `d_ff`; `next_token_logits`
on the padded buffer matches `forward` on the exact-length sequence to **6e-8**, so the
fixed-shape trick is genuinely equivalent; nucleus sampling matches a reference
implementation across five `top_p` values; train and val cannot overlap and no window
crosses the boundary; an AST pass over every code cell found **no cell using a name defined
in a later cell**; and an AST free-variable pass over the generated `nanochat_model.py`
found **no missing name** — the module imports clean and round-trips in a fresh process.
Also confirmed: QK-norm after RoPE commutes with QK-norm before it (RoPE is norm-preserving,
RMS-norm is a scalar rescale), so applying it in the other order from the reference is not a
difference.

| # | Sev | Bug | Fix |
|---|---|---|---|
| B1 | S2 | `generate` returned `''` with no error for any prompt of ≥ `seq_len` tokens — truncation kept `seq_len`, leaving no room to write | keep `seq_len - 1` |
| **B2** | **S3→real** | **`chars_per_token` counted `<\|endoftext\|>` as 13 characters of story**, inflating compression 2.1037 → 2.1382 (**+1.6%**) and understating bpc by the same | drop separators from the character count, keep them in the token count |
| B3 | S3 | `sample_batch` upper bound one short (`maxval` is exclusive); one legal window unreachable | `len(data) - context_len` |
| B4 | S3 | final `save_checkpoint` used the loop variable `step`, undefined if the loop never runs (a `RESUME` past `n_steps`) | initialise `step = start_step` |
| B5 | S3 | **resume replayed the data** — the PRNG key reset to seed 0 while the step counter continued, and the key was not checkpointed | draw batch *t* from `fold_in(root_key, t)` |
| B6 | S3 | `save_checkpoint` raised on a bare filename (`os.makedirs('')`) — never hit here, but the function ships in `nanochat_model.py` | guard the empty dirname |
| B7 | S3 | prose claimed the char baselines were computed "on this corpus"; they are on the **valid** split (vocab 91) while the model is scored on held-out **train** text (vocab 104) | name the split, state the caveat |

**B2 is the one that mattered**, and it is the same failure mode WP5's review found: a
number that looks right, is quietly wrong in the flattering direction, and would have been
published. The fix moved the rehearsal's bpc from 1.661 to **1.680** — *worse*, which is the
tell. Worth noting the reviewer's proposed fix (drop separators from both counts) was itself
slightly wrong: the model spends bits predicting the separator, so those bits have to be
charged against the real text. Take a reviewer's diagnosis more readily than its patch.

**B5 is the one worth remembering.** "Reproducible" and "resumable" are different
properties. A running `split` gives the first, not the second; `fold_in(root_key, step)`
gives both, and costs nothing.

### N4 — adversarial fact-check of WP6 · 2026-09-26 · **the ±13% was mine, and it was wrong**

Second review dimension: every number in the prose checked against the notebook's own
executed output. All nine fixes are **markdown-only**, so no output was touched and no
re-run was needed — applied through the `nbformat` API, because at 539 KB the notebook is
past what `Read` will open and `NotebookEdit` therefore cannot reach it (the failure mode
`CLAUDE.md` documents). Validated afterwards: `nbformat.validate` clean, cell count and
types unchanged, and **every code cell's source, outputs and `execution_count` asserted
byte-identical** to before the edit.

**Confirmed correct** — a long list, which is the point of asking: 104+919+1 = 1024; the
26,223,104 parameter count derived independently from the shapes; the step-budget
derivation 20 × 26,223,104 / 8192 → 64,000; 524M tokens = 0.51 epochs; bpt = 0.8087/ln2 =
1.1667 and bpc = 1.1667/2.1035 = 0.5546; uniform bpc = 10/2.1035 = 4.754; the 4 GB figure
for the materialised windows (4,039,642 × 257 × 4 B = 4.15 GB); the 1.64% separator
inflation; RoPE θ_i and the split-half convention; and the deliberate absence of any numeric
*prediction* for the post-fix entropy ratio, so the measured 0.687 contradicts nothing.

| # | Sev | Wrong claim | Correct |
|---|---|---|---|
| **1** | **HIGH** | "every weight sits within about **±13%** of 1/256", stated in **two** cells and load-bearing for Exercise 4 | **+28.3% / −22.1%.** Weights lie in [0.78, 1.28] × 1/256 |
| **2** | **HIGH** | "4.6 s including the compile, then 0.2 s — about 3 ms/token" | The cell directly below prints **5.2 s, then 0.5–2.4 s**. Prose now points at the printed timings and explains the spread |
| 3 | MED | embedding+head is "the *dominant* term for a small model" | 4.0% here, 14.3% at depth 4; dominant only above V ≈ 6,000 |
| 4 | MED-LOW | int64 corpus "8.5 GB" | **8.31 GB** (the 8.5 was the *old* windowed file, not this one) |
| 5 | MED-LOW | "bpc ≈ 1.0–1.5" next to a measured 0.555 | true of *general* English; TinyStories is far more predictable — now said |
| 6 | LOW | entropy table's three columns do not divide | ratio column is the **mean of per-row ratios**; Jensen. Explained in the markdown rather than changing a printed output |
| 7 | LOW | "generated by GPT-3.5 and GPT-4" | V2-GPT4 is **GPT-4 only** |
| 8 | LOW | dead link `bpe_tokenizer.ipynb` | `bpe-tokenizer.ipynb` |
| 9 | COSMETIC | shape check prints `= 4.000` two cells after the prose argues the value is 8 | toy `head_dim` is 16; noted in prose |

**Finding 1 is the one to learn from.** ±13% is `e^0.125 − 1` — the deviation from the
*geometric mean* of the two extremes, not from uniform. The correct bound comes from the
softmax itself: one logit at +0.125 against 255 at −0.125 gives
`e^0.125 / (e^0.125 + 255·e^-0.125) = 1.283 × 1/256`. I derived a plausible-looking number
from the right starting point and never checked it against the definition, then repeated it
in a second cell and built an exercise on it. **The conclusion was unaffected** — a spread
of 0.78–1.28× still forces an entropy ratio of ~0.997, which is what the A/B measured at
0.9995 — which is exactly why it survived two passes: the story it supported was true.

**Both S1-severity findings across N3 and N4 were numbers that flattered or simplified in a
believable direction** (the bpc inflation, and this). Neither broke anything. That is the
class of error this project keeps producing, and the only thing that catches it is a
reviewer told to check arithmetic against the definition rather than against the narrative.

## N5 — does the reference's fixed `1.2` matter? · 2026-09-26 · **the answer is a better lesson than the answer**

**Question.** The nanochat reference applies RMS norm to Q and K and *then* multiplies both
by a hard-coded `1.2`. WP6 shipped without it, on the reasoning that it "only rescales an
already-healthy logit range". Yoav asked for that reasoning to be tested rather than
asserted.

**It is not only a rescale of the ceiling.** Measured at initialisation, before any training:
attention-logit std **1.0022 without, 1.4431 with** — a ratio of 1.4399, i.e. exactly the
1.2² = 1.44 the algebra predicts. So it is a fixed temperature on the attention softmax, and
the original justification was wrong about the mechanism even though it may be right about
the outcome.

**Design.** As N1: same script, one line different, everything else identical. Three seeds
(0, 1, 2) per variant this time — N1's single-seed rule was justified by runs costing hours,
and these cost nine minutes, so the exemption did not apply. `l2` was left at one seed
because its gap is ~200× the wobble. Six runs, ~30 min on two A4000s.

| variant | n | val @ 4,000 | sd | entropy ratio | init logit std |
|---|---|---|---|---|---|
| `l2` | 1 | 1.5027 | — | 0.9995 | — |
| `rms` | 3 | 1.0738 | 0.0047 | 0.666 | 1.0022 |
| `rms12` | 3 | **1.0678** | **0.0008** | **0.530** | 1.4431 |

**Δ(rms → rms12) = 0.0060 nats, Welch t = 2.19, df = 2.1, p = 0.15.**

**The result is a genuine tension, and it is now the notebook's teaching material.**
`openai/parameter-golf` accepts a new record only if it beats the old by **≥ 0.005 nats**
*and* the logs show that at **p < 0.01**. Our gap is 0.0060 — it **clears the magnitude bar
and fails the evidence bar by more than an order of magnitude**. The two halves of one
published standard disagree about the same measurement. Nothing here resolves that, and the
notebook deliberately does not try to.

**The accidental finding is the most useful one.** Between N1's `rms` run and this one, the
only edit to the script was multiplying q and k by a constant equal to `1.0` —
*mathematically a no-op*. The two runs finished at **1.0707 and 1.0686, 0.0021 apart**, with
a step-by-step wobble of ±0.003 from step 200 onward. A null code change moved the answer by
a third of the effect being measured. That is the noise floor, and it was free.

**Also worth noting, and unexplained:** `rms12`'s seed sd is 0.0008 against `rms`'s 0.0047 —
roughly 6× more stable across seeds. Three runs is far too few to call that real; it is
recorded here because it was measured, not because it is claimed.

**Decision: unchanged — WP6 ships without the 1.2**, and the notebook's prose no longer
claims it is a mere rescale. The evidence does not meet the standard the notebook itself
now cites, and reversing would cost a 2.1 h retrain plus WP7/WP8 checkpoint regeneration.
Reversible if Yoav prefers the reference exactly.

### The notebook now carries a second story (Yoav, 2026-09-26)

Overriding "one story per notebook" for `nanochat.ipynb` only: an *algorithmic* story (how a
GPT works and is trained) and a *data-science* story (many hyperparameters, unexpected
sensitivity, so how do you decide?). Delivered as a **demonstration that opens a discussion**
— the questions are posed and the conclusions deliberately left unwritten, because they are
the classroom's. Two new cells after the bits-per-character section, reading
`checkpoints/qknorm_ab.json`, which commits all six runs.

**Three mechanical notes on how it was landed**, each of which cost something:

1. **The demonstration's code cell was executed on its own, not as part of the training run**,
   and carries `execution_count: null` to make that visible. It is real output from that exact
   source against the committed JSON — it reads a file and prints, so it depends on no kernel
   state. Re-executing the whole notebook to renumber it would cost 2.1 h *and* shift every
   published number by ~0.002 through the GPU nondeterminism this very section is about.
2. **Source size was the binding constraint.** The section was paid for by trimming the
   SFT/GRPO preview (which duplicated the two notebooks that own that material, 3,023 → 1,147
   chars) and then by tightening the new prose repeatedly. Final source: **24,994 tokens, 6
   under the 25,000 limit.** The next edit to this notebook must either cut source first or
   go through `nbformat`.
3. **All edits went through the `nbformat` API**, since at 539 KB the notebook is far past
   what `Read` will open. Every pass asserted the untouched cells' source and outputs
   byte-identical afterwards.

### N6 — review of the restructured notebook · 2026-09-26 · **a correction I had asserted but not made**

Third review of WP6, after the SFT/GRPO removal, the exercises move and the prose trims.
Eight findings, three S1.

**Confirmed correct**, which is why the restructuring was cheap to verify: the notebook uses
**no numeric in-text citations at all** — every reference is an inline link — so deleting
DeepSeekMath and renumbering broke nothing; only one `exercis` hit remains (the pointer to
the new notebook); every A/B number in the prose matches the cell beneath it; the Welch
statistics recompute (t = 2.18, df ≈ 2.1, p ≈ 0.15); the attention arithmetic survived the
compression; and the exercises' import list was cross-checked against `MODULE_FUNCTIONS`.

**The finding that matters is a process failure, not a content one.** The previous commit
message, `runs.md` N5, and my report to Yoav all stated that the notebook's prose "no longer
claims the 1.2 is a mere rescale". **It still did — word for word — three cells above the
section built to disprove it.** I wrote the claim into the log as done, then cited the
experiment that refutes it, and never grepped. This is the *same* failure the 2026-09-26
decision "a correction is not landed until you have grepped for it" was created from, one
package later. The rule is not enough on its own; what would have caught it is grepping for
the **old string** rather than confirming the new one exists.

**The second S1 is the notebook's own lesson catching the notebook.** Question 4 claimed our
0.0060-nat gap "clears" parameter-golf's 0.005 threshold. It does per *token*; their metric
is per *byte*, and at 2.104 chars/token the gap is **0.0029 per character**, which fails.
The bits-per-character section one page earlier exists precisely to warn against comparing
per-token numbers across tokenisers. The question now presents both axes and asks which is
fair — a better exercise than the tidy version, and it is left unresolved like the rest.

| # | Sev | Finding |
|---|-----|---------|
| 1 | S2 | A paragraph duplicated verbatim in the QK-norm cell — an insert/replace scar from the `nbformat` passes |
| **2** | **S1** | **The stale "only rescales an already-healthy logit range" claim** — see above |
| 3 | S1 | The Henry et al. caveat sat *before* the paragraph it critiques, and its "the learned gain is what restores the dynamic range" contradicted "there is no learned gain here" three paragraphs later. Resolved by introducing RMS as reaching the same end by a *fixed* route |
| 4 | S1 | Exercises notebook: "everything you need is importable", then an import list missing `sample_batch`, `train_step`, `validate` and the optimiser — none of which are in `nanochat_model.py`, and five of seven exercises train something. Invisible while the exercises lived inside the notebook |
| 5 | S2 | One noise floor quoted (0.002, bit-identical re-run) where there are two; the seed-to-seed figure is **0.005** and is the bar that matters. Exercise 4 also asked students to predict a number the notebook now prints — reframed as a reproduction at half the budget |
| 6 | S2 | The unit mismatch — see above |
| 7 | S3 | "the two `rms` runs below" — only one of the pair is in the table; the other is from N1 |
| 8 | S3 | `init_logit_std` not printed, so question 2's mechanistic claim was the one thing unverifiable from the output; "a clear improvement on 128" unsupported; "same corpus" overstated; Henry et al. missing from the reference list |

**Two structural notes acted on:** the notebook had **two endings** (the A/B section read as a
close, then module housekeeping, then the comparison table) — the A/B section now sits after
the module cells. And entropy 0.666 (4,000 steps) against 0.687 (64,000) is now connected in
one clause, rather than left to look like a typo.

**Left undone deliberately:** the review wanted `nanochat-sft.ipynb` and `nanochat-grpo.ipynb`
linked as a forwarding address for the deleted section. Removing that material was the
instruction; `index.ipynb` lists them.

### On the A/B harness, for whoever needs one next

`qknorm_ab.py` is **not committed and is gone with the session that ran it.** That is
deliberate, not an oversight: it was written before `nanochat_model.py` existed, so it
re-pastes the model, the optimiser and the training loop — a fifth copy of exactly the thing
WP6's code-reuse contract removed. Resurrecting it would reintroduce the drift.

Rebuild rather than recover. The evidence it produced is committed
(`checkpoints/qknorm_ab.json`, six runs), so nothing is lost but ~150 lines of scaffolding,
and the replacement is shorter and better:

```python
from nanochat_model import init_params, forward, cross_entropy_loss, precompute_rope, causal_mask
```

then copy `sample_batch`, `train_step` and `validate` from `nanochat.ipynb`, and vary the one
thing under test. What is worth carrying over from the original, because each was learned the
hard way:

- **Hold the validation batches fixed across every variant and seed** (a constant key), so the
  curve reflects the model changing and not the sample changing.
- **Seed the init and the data order together**, and write one JSON per run — aggregate after.
- **Measure the mechanism, not only the loss.** The entropy ratio and the init-time logit sd
  are what made N1 and N5 explanatory instead of merely conclusive.
- **Run the variants concurrently, one per GPU.** Two variants cost one variant's wall-clock.
- **Budget three seeds** whenever the expected effect is near the 0.005 bar; one seed is only
  defensible when the effect is ~200x the noise, as in `l2` vs `rms`.
---

## S1 — SFT on TinyStories-Instruct · 2026-09-27 · **the task swap worked: 0.092 → 0.467 constraint satisfaction**

WP7's committed run. New ID prefix `S` for the SFT notebook; `N*` is the pretraining series.

The package's premise was that `nanochat-sft.ipynb`'s old task — continue a story from its
first sentence — could not demonstrate anything, because the pretrained model already does
it fluently (review finding 9.1). Replacing it with constrained generation from
TinyStories-Instruct was supposed to make the before/after visible. It does.

**The premise, verified before any code was written.** Prompted with
`<|endoftext|>Words: sad, jump, big\nFeatures: Dialogue, BadEnding\nStory:\n`, the pretrained
checkpoint writes dialogue in which **`Words` and `Bad` are character names** ("So, Words,
Bad and Papa all stood in the same..."). It does not read the headers as instructions at
all. That failure is cell 9 of the notebook, generated before training so it cannot be
re-shot to suit the conclusion.

| | |
|---|---|
| dataset | `TinyStories-Instruct-valid.txt`, 26.9 MB, 25,028 records |
| usable | 17,466 have the instruction-then-story shape; **11,620 examples** after requiring `Words:` |
| dropped | 5,844 no `Words:`; **2** for a character outside the closed vocabulary; 0 for not fitting |
| build cost | **6.7 min** of pure-Python BPE, cached to `checkpoints/sft_examples.npz` |
| shape | prompt mean 34 tokens, response mean 209 (min 100, max 237, 90% in 186–229) |
| split | last **1,024** examples held out, fixed |
| budget | 6 epochs = **1,986 steps** × 32, lr 3e-5, no early stopping |
| wall-clock | **4.6 min** training on one A4000, plus 7.9 min for 768 evaluation generations |

**Held-out response loss 0.8437 → 0.6533** (best, step 1,000 = 3.0 epochs), improvement
**0.1903** against the 0.005-nat noise floor from N1/N2. The last step (1,986) ends at
0.6639, **+0.0105 worse than the best** — the budget deliberately overshoots so the curve
turns inside the committed plot.

**Constraint satisfaction** (n = 256 held-out prompts, temperature 0.8, shared sampling
keys; ± is one standard error over prompts):

| model | words found | all three | tokens |
|---|---|---|---|
| pretrained | 0.092 ±0.011 | 0.004 | 210 |
| **SFT, best (step 1,000)** | **0.467 ±0.018** | 0.109 | 207 |
| SFT, last (step 1,986, overfit) | 0.490 ±0.017 | 0.113 | 208 |
| ground truth (ceiling) | 0.686 ±0.017 | 0.332 | 207 |

**No forgetting at this learning rate.** Plain next-token loss on held-out pretraining text
went **0.8340 → 0.8247** — slightly *better*. Not a general result; it says `3e-5` was
conservative and the domains overlapped.

### The budget question, answered by experiment (the S2 probe below)

Yoav asked whether a longer SFT — "30 min?" — would help. It would not, and the run that
settles it is worth more than the answer.

### Three results that are more interesting than the headline

**1. The two metrics disagree about overfitting.** Held-out loss says the last-step
checkpoint is clearly worse (+0.0105, twenty times the noise floor). Constraint satisfaction
says it is *slightly better* (0.490 vs 0.467, about 1.3 standard errors — i.e. not
reliably anything). So the quantity being optimised degraded while the behaviour we actually
want sat still. **Watching only the behavioural number would have concluded six epochs was
fine.** This is now the notebook's argument for why best-checkpoint saving is by loss:
loss is measured on all 1,024 examples and resolves 0.001; the behavioural metric at n=256
cannot resolve 0.03. Precision is a reason to prefer a metric, not only relevance.

**2. The word matcher is stricter than it looks, and it is WP8's reward.** `words_present`
matches on word boundaries, so `jump` does not match "jumping". That is why the ground-truth
ceiling is 0.686 and not the ~0.98 a substring test gives. The comparison stays fair — the
ceiling is scored under the same rule — but **WP8 must revisit this before optimising against
it**, because a policy turns every weakness in a matcher into a strategy. Substring matching
is not the fix either: it counts `war` inside "warm". Stemming, or a shared-prefix rule, is
the place to start; re-measure the ceiling under whatever rule is chosen.

**3. The coverage metric is not reproducible to better than ~0.013 between full runs, and I
could not explain why.** Two executions of the notebook whose *computational* code was
byte-identical (only a code comment and two markdown cells differed) produced
pretrained 0.094/0.092, SFT-best 0.454/0.467, SFT-last 0.488/0.490 — while the ground-truth
row (pure data) was identical, the training log was bit-identical apart from wall-clock
seconds, and the loss table matched to four decimals. Isolated generation **is** reproducible:
the same 64 prompts from the same checkpoint with the same keys gave byte-identical output
twice within one process and again in a second process. So the variation appears only in the
full-notebook path, where training runs first; the likeliest cause is XLA autotuning state,
but **this is a hypothesis, not a measured conclusion.** Practical consequence: the printed
± understates total uncertainty by roughly a factor of √2, and any WP8 reward comparison
needs a run-to-run floor established the same way N1 established the loss floor.

### Method notes worth carrying forward

**Rehearse first, execute once.** Every code path was tried on a 900-example subset (27
steps, 16 generations, plot, checkpoint round-trip) before the real run — two minutes, and it
confirmed the signal was real (coverage 0.083 → 0.208 after 27 steps) before committing to a
20-minute execution. WP6's lesson was "review *during* the run, because `nbconvert --inplace`
overwrites edits made while it is going"; this is the cheaper form of it.

**Estimate the executed file size before running, not after.** This package burned three
executions on notebook size. The arithmetic that would have avoided it:
`(source chars + expected text output + ~30,000 for one figure) / 3.4` ≈ tokens, where 3.4
chars/token is measured, not assumed. Two corrections to what `CLAUDE.md` currently implies:
a figure costs ~9,000 tokens and **its cost is dominated by physical dimensions** — dpi and
point count barely move it (downsampling a 1,962-point line to 199 saved 4 KB of 38 KB) —
and, more importantly, **the executed size is not the constraint that matters.** Every
notebook in this repo exceeds the 25k `Read` limit when executed (`nanochat.ipynb` 152k
tokens, `sets.ipynb` 72k, `GRU.ipynb` 26k); what keeps a notebook editable is its **source**
size, because the way back is always clear-outputs → edit → re-execute. This notebook's
source is 15.0k tokens; its executed form is 26.3k, the second smallest in the repo.

**A watcher shell built as `until ! pgrep -f "nbconvert.*X"; do sleep; done` matches its own
command line** and waits forever, reporting a finished run as still going. Cost here: a run
that completed at 00:50 was still being reported as running at 07:14. Use a sentinel file
(`echo $? > DONE`) instead.

---

## S2 — how long should SFT run? · 2026-09-27 · **validation bottoms at 3 epochs; 15 epochs buys nothing**

Yoav asked whether to run SFT for ~30 minutes instead of ~3. Rather than guess, a standalone
probe trained the same setup for **15 epochs (4,965 steps, 15.0 min on one A4000)**, scoring
the full 1,024-example validation set every 100 steps and constraint satisfaction (n=64)
every 993.

| steps | epochs | train | val | words found | all three |
|---|---|---|---|---|---|
| 993 | 3.0 | 0.582 | 0.6533 *(minimum at step 1,000)* | 0.427 | 0.094 |
| 1,986 | 6.0 | 0.486 | 0.6641 | 0.427 | 0.062 |
| 2,979 | 9.0 | 0.435 | 0.6741 | 0.411 | 0.062 |
| 3,972 | 12.0 | — | — | 0.479 | 0.078 |
| 4,965 | 15.0 | — | — | 0.479 | 0.109 |

**Answer: no.** Validation loss reaches its minimum of **0.6533 at step 1,000** and rises
monotonically thereafter, while training loss keeps falling (0.582 → 0.426 by step 2,400).
Thirty minutes would be ~10,000 steps, roughly ten times past the optimum.

**And constraint satisfaction does not improve either** — it wanders between 0.411 and 0.479
across a five-fold range of training, which at n=64 (SE ≈ 0.04) is flat. My first reading of
this table called the all-three column a *degradation* (0.094 → 0.062); the fourth and fifth
points (0.078, 0.109) show that was noise, and the claim was withdrawn before it reached the
notebook. **Two points are not a trend when the standard error is 0.04.**

That flatness is the useful part, and it is the strongest available argument for GRPO: past
the loss minimum, more imitation buys neither likelihood nor behaviour. The gap to the 0.686
ceiling is not a budget problem.

**What changed in the notebook as a result:** the budget went from 3 epochs to **6** — not to
train better, but so the committed plot shows the turn. Three epochs left the curve still
falling, which made best-checkpoint saving look like dead machinery (best = last step) and
forced the prose to admit it had never seen the overfitting it was sized to avoid. Six epochs
puts the minimum mid-plot at a cost of ~3 minutes, and produces the best/last comparison that
finding 1 in S1 rests on.

**Reproducibility note:** the probe reproduced the notebook's validation curve step for step
(0.6948 @100, 0.6772 @200, 0.6686 @300, … 0.6533 @1,000) from a separate process and script,
which is a stronger determinism check on the training path than anything in S1.

---

## G1 — GRPO on word-constraint satisfaction (WP8) · 2026-09-27 · **the reward went 0.46 → 0.94 past a 0.82 ceiling: a real reward hack, kept**

WP8's committed run. New ID prefix `G` for the GRPO notebook; `S*` is SFT, `N*` pretraining.
(Note for whoever indexes this file: `S1` is used twice — line ~416 for `sets.ipynb` under
WP4, and line ~1194 for SFT under WP7. Not renumbered here, but do not cite "S1" unqualified.)

Everything below is on one A4000, starting from `nanochat_sft_best.pkl` (step 1,000,
val 0.6533), prompts drawn from the 10,596 SFT training examples, evaluation on the 128
held-out prompts SFT never trained on either.

### The headline

| | words found | all three |
|---|---|---|
| chance (SFT text, wrong words) | 0.022 ±0.006 | 0.000 |
| pretrained | 0.119 ±0.009 | 0.000 |
| SFT — where GRPO starts | 0.459 ±0.015 | 0.074 |
| **GRPO, 150 steps** | **0.943 ±0.007** | **0.836** |
| ground truth (ceiling) | 0.820 ±0.020 | 0.562 |

Paired over the same prompts and the same sampling keys: **+0.484 ± 0.016**, 30.4 standard
errors, 98% of prompts improved and none got worse. 150 steps = 4,800 generations in
**32.1 min**; the whole notebook is ~40 min, about 40% of it evaluation.

**And it is a reward hack**, which is why the run was kept rather than retuned. The policy
scores 0.126 *above* the stories the corpus itself provides. It gets there by using the three
required words **8.59 times in total per completion** against **4.38** in the real stories
and 2.76 under SFT, with a lower distinct-token ratio (0.582 vs 0.659 for ground truth). Three named
strategies, all visible in positionally-chosen samples:

- inserting the word regardless of sense — asked for `jump, value, ashamed`: *"Tim was a
  very ashamed frog. He did not like to jump high in the sky like the value of his
  friends."*; and for `put, cliff, glad`: *"a glad cliff was done."*
- repeating until something lands — *"Tim jumped over a small pond to jump"*, *"He put more
  and more cliffs on the cliff."*
- and, on the third prompt, something that mostly reads like a story — *"there was a busy ant
  named Andy. Andy wanted to sign his name to see the infant"* — which still repeats `infant`
  four times, and which scores **1.00 exactly like the other two**. The reward cannot rank
  them; that comparison is the sharpest thing in the notebook.

**Three corrections to earlier drafts of this entry, and the pattern is the point.** (i) It
quoted a *different* run's generations, because re-executing retrains the policy. (ii)
Rewritten against the committed output, it still mischaracterised two of them — calling a
passage "naming a character after the required word" when the required words were `jump,
value, ashamed` and the character was Fred; and offering *"he saw a valuable frog"* as the
policy gaming the matcher when `valuable` is not in `word_forms('value')` and earns zero.
(iii) It then went stale **again** when runs 5 and 6 retrained the policy, and was caught by a
review, in the very entry that records (i) and (ii). A quotation being verbatim does not make
the sentence around it true, and prose about generated text has to be re-derived after *every*
execution — there is no version of this that survives a re-run untouched.

**The KL term did not stop it.** β = 0.04 (the DeepSeekMath value) held drift to ~0.059
nats/token, and that was enough. The forgetting check prices the damage: held-out
language-modelling loss **0.8247 after SFT → 0.8384 after GRPO**, worse than the *pretrained*
model's 0.8340. The RL stage spent SFT's fluency gain, and some of pretraining's, buying
reward.

### Why this is the better artifact

Measurement rigour was necessary and insufficient. The fixed prompt set, shared keys,
clustered SE and paired difference all did their jobs — the effect is real, reproducible and
enormous. They bought *precision about the wrong quantity*. Two things caught the problem and
neither is a statistic about the reward: **the ceiling row**, measured before any training
(§2 fixes the rule and scores all 1,024 held-out stories at 0.837; §6 scores the 128
evaluation prompts at 0.820, the number the result is compared against), and **reading the
output**. That is now the notebook's §10.

### Measured facts worth not rediscovering

**The evaluation is bit-reproducible, within a process and across processes.** Two scorings
of one policy returned identical per-completion scores on all 512 samples, and a second
process reproduced the first exactly (mean 0.4590 both times, `max |diff| = 0.0000`). This
closes the question WP8's plan entry raised as a blocker ("an RL improvement smaller than
~0.03 is not evidence until that floor is measured"). **The floor is zero** under this
protocol — enumerated prompts, keys derived from batch position, and one fixed-shape compiled
sampler — so the only uncertainty is the sampling SE the table prints (±0.007–0.020). WP7's
0.013 drift did *not* reproduce here; the cause there was never identified and is not
re-opened, but nothing in this path exhibits it. A reduced version of the check now runs
inside the notebook and asserts, so the claim cannot silently rot.

**Batched, scanned sampling is ~34× faster than the old loop.** Fixed-width buffer, all B
sequences in one call, token loop inside `jax.lax.scan`: **206 ms per 194-token generation**
at B=32, against ~4.4 s for the previous per-token, batch-1, `.item()`-syncing version. That
is what turned the plan's projected 8–12 h into ~40 min. `grpo_loss` padded and jitted costs
0.27 s per gradient step at B=32 — about 2% of a step, exactly as the plan predicted, which
is why sampling was fixed first.

**K > 1 is what makes clipping exist.** At K=1 the clipped fraction is exactly 0.0000 by
construction (ρ ≡ 1). Measured after all 4 inner epochs: **0.0090** mean over the run, rising
to ~0.066 on the first step. Small, but non-zero and real.

**Dead groups rise as the task is solved**: 0% early, **39.2% over the whole run**, because
groups increasingly score all-1.0. A high dead fraction late is not a bug, it is the signal
that the reward has been saturated.

**Ground-truth ceiling depends on the matcher, so it must be re-measured with it.** Under the
inflection matcher: 0.837 / 0.622 all-three on the 1,024 held-out stories. Under
`nanochat-sft.ipynb`'s exact-spelling rule: 0.768 / 0.470. The strict rule reproduced WP7's
committed 0.686 exactly on its first 256 prompts, which validated the pipeline before
anything was changed.

**Ground-truth responses average 209 tokens**, so the old `max_new=80` was measuring
truncation. The budget is now 194 — one constant set by the longest prompt (61 tokens) so the
compiled program has one shape. Residual known limitation: 83% of eval prompts have
ground-truth stories longer than 194, so the comparison is still not budget-neutral, only far
closer than before.

### Process notes

**Two adversarial reviews, dispatched at the start of the run, found 26 issues between them**
— and the run was killed at ~10 min rather than spend 45 on numbers that would be discarded.
The load-bearing ones:

- **The clipping metric was measured before the update.** `jax.value_and_grad`'s aux
  describes the parameters the gradient was taken *at*, so the "after epoch 1" curve was
  identically 0.0000 *at every K* — a tautology being plotted as a measurement — and the
  "after epoch K" curve actually showed K−1 updates. An exercise had been built on it. Fixed
  with a metrics-only jit evaluated after the inner loop (one extra forward per step).
- **The reward had reward-hacking surfaces of its own.** `word_forms` generated `-er`/`-est`/
  `-ly` and an ungated `-es`, producing real words that are not forms of the required word:
  `let`→letter, `corn`→corner, `man`→manner, `moth`→mother, `mat`→mates, `on`→ones. Dropped
  and gated. Separately the tokeniser lost `wife's` as a use of `wife`, which made four
  stories score *lower* under the "looser" matcher than the strict one — invisible in the
  averages, which moved the expected way regardless. Now a runtime assertion.
- **Dr. GRPO was misattributed.** The notebook credited the *std* divisor with inflating
  response length. Liu et al. §3 is explicit: dividing by `|o_i|` gives the response-length
  bias, dividing by `std(R)` gives a *question-difficulty* bias. Verified against the PDF,
  not from memory. The quoted phrase in reference 3 is from the paper's Figure 1 caption —
  a reviewer flagged it as unsourceable having checked only the abstract and README.
- **"DeepSeek-R1-Zero starts from a 7B base model"** — false, inherited verbatim from the old
  notebook. It is DeepSeek-V3-Base, 671B MoE.
- Several claims about the *old* notebook were overstated and were corrected rather than
  dropped: early stopping **never fired**; the headline SFT-vs-GRPO comparison **did** share
  sampling keys (both defaulted to `seed=7`); the reported difference was +0.112, not ~0.03.

**Rehearsing at tiny scale paid for itself three times.** The first rehearsal died instantly
on `ModuleNotFoundError` because the throwaway copy sat outside the repo — `nbconvert` runs
the kernel in the notebook's directory. A later one exposed a figure panel plotting KL (~1e-3)
against a 0–1 fraction on one axis. Both would have cost a full run.

**A watcher built as `while pgrep -f "<pattern>"` matches its own command line** and never
exits — the same trap W6 recorded, hit again in a different form. Waiting on an explicit PID
(`while kill -0 $PID`) works.

### G1 addendum — how this compares with `karpathy/nanochat`'s RL stage

Checked against the source 2026-09-27 (`scripts/chat_rl.py`, `tasks/gsm8k.py`), because the
notebook cites nanochat as the implementation this series follows and was silently differing
from it on every axis.

**The reward is the important difference, and it explains the hack.**

| | karpathy/nanochat | this notebook |
|---|---|---|
| task | GSM8K word problems | write a story using three given words |
| reward | `float(is_correct)` — binary, exact match on the extracted final numeric answer | fraction of the three required words present |
| gameable by surface manipulation? | **no** | **yes** |

GSM8K's reward scores a **verifiable outcome**: one correct number, extracted and compared.
Repetition, renaming and padding cannot move it. Ours scores a **surface feature** — does this
token appear — and surface features are exactly what a language model can manipulate without
doing the task. The reward hack in G1 is therefore not bad luck; it is the predictable
consequence of paying for a proxy instead of an outcome. Worth stating in any future package
that adds a reward: *ask whether the reward can be satisfied without doing the task, before
running anything.*

**And karpathy's RL is deliberately not GRPO.** Verbatim from the header of `chat_rl.py`:

> I put GRPO in quotes because we actually end up with something a lot simpler and more
> similar to just REINFORCE:
> 1) Delete trust region, so there is no KL regularization to a reference model
> 2) We are on policy, so there's no need for PPO ratio+clip.
> 3) We use DAPO style normalization that is token-level, not sequence-level.
> 4) Instead of z-score normalization (r - mu)/sigma, only use (r - mu) as the advantage.

Where this notebook stands against those four:

| | karpathy | this notebook |
|---|---|---|
| 1. KL to reference | none | **has one** (β = 0.04) — and it did not prevent the hack |
| 2. PPO ratio + clip | none (on-policy, K=1) | **has it**, and K=4 is what makes it bind at all |
| 3. normalization | token-level (DAPO) | **token-level — same choice** (`response_mask.sum()` batch-wide, not per sequence) |
| 4. advantage | `r - mu` | `(r - mu)/sigma` — **differs** |

Two consequences worth carrying forward. (a) On point 3 the notebook already agrees with both
karpathy and Dr. GRPO, which is why §10's claim that it "carries only the standard-deviation
half" of the GRPO bias is correct. (b) On point 4, Exercise 4 asks the student to drop the
sigma — which converges on karpathy's choice *and* Liu et al.'s recommendation simultaneously.
That was coincidence when the exercise was written and is now stated as the point of it.

Keeping the KL and the clip is a deliberate divergence, not an oversight: they are the
subject being taught, and a notebook that deleted them would have nothing to say about
clipping or trust regions. The honest framing — now in the notebook — is that karpathy's
simplifications are what you reach for when you are on-policy and want the thing to work,
and the full objective is what you implement when you want to understand what was simplified.

---

## G2 — does a fluency term close the exploit? · 2026-09-27 · **no: pretrained likelihood prefers the hacked text to the real stories**

Yoav asked whether a different reward avoids G1's failure. Rather than argue, a standalone
probe (`scratchpad/probe_reward.py`) ran the committed notebook's own cells — same matcher,
sampler, rollout and loss, byte-identical — and swapped only the reward:

```
reward = words_present(text) * fluency(text)
fluency = clip((HI - nll) / (HI - LO), 0, 1)
```

`nll` is the completion's per-token negative log-likelihood under the **frozen pretrained
model** — a reward model we already had and never trained. Multiplicative on purpose, so the
two factors cannot be traded against each other. `LO`/`HI` are the 50th and 99th percentiles
of the *real stories'* own NLL, so the bar is set by the corpus rather than invented.

### It did not work

| | words found | fluency | composite | all three |
|---|---|---|---|---|
| SFT | 0.459 | 0.995 | 0.456 | 0.074 |
| **GRPO, composite reward** | **0.954 ±0.007** | 0.990 | 0.944 | 0.867 |
| ground truth (ceiling) | 0.820 | **0.707** | — | 0.562 |

150 steps, 25.1 min. G1 (words-only) reached 0.943; G2 reached **0.954** — no better, and
still 0.134 above the ceiling. Required-word mentions 8.53 against ground truth's 4.38
(G1: 8.59); distinct/token 0.595 against 0.659 (G1: 0.582). The gate never engaged.

### Why, and the number that predicts it before you train

**The pretrained model assigns higher likelihood to the policies' own text than to human
writing**: fluency 0.99 for both SFT and GRPO completions, **0.707** for the real stories.
That is not a tuning failure, it is the wrong signal — repetition is high-likelihood, which
is the well-known neural text degeneration result. An anti-degeneracy term built out of
likelihood is being asked to penalise exactly the thing likelihood rewards.

The calibration alone says this, in ~20 seconds, before a single training step:
real-story NLL p50 **0.8355**, p90 1.0896, p99 1.2891, against SFT completions sitting near
the top of that range. **Run the calibration before the 25 minutes of RL.**

Sample, scoring well: *"Kitty had a value to her friends … let's jump higher than us find
value to each other!"* — ungrammatical insertion, high likelihood, paid in full.

### What to try instead

Price the measured symptom, not a proxy for it, and keep the multiplicative structure:

```
reward = words_present(text) * mention_gate(text)     # thresholds from the corpus:
                                                      # ground truth 4.38 mentions, hacked 8.59
```

or a distinct-token floor (ground truth 0.659, hacked 0.582). Both are *also* surface
statistics and therefore also gameable — mention each word once and pad with varied filler —
so the honest framing is that at 26 M parameters you choose which failure you can live with.
The general answer is a learned reward model or human evaluation, which is where production
RLHF actually spends its effort.

**Second, independent reason the composite structure is right:** `words_present` *saturates*.
Once the policy reliably lands all three words there is no gradient left, which is what G1's
39.2% dead-group fraction is. A reward with a continuous quality term cannot saturate.

### What goes in the notebook

The calibration, not the training run — it is the stronger argument (a property of the
signal, not of one run) and costs 20 seconds against 25 minutes. G2's outcome is cited here
in one line rather than re-run in the notebook, which keeps it at ~40 min.
