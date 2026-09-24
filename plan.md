# nanochat revision plan

Tracking document for the revision described in `nanochat-review.md` (v2, 2026-09-20).
Finding IDs (`8.3`, `A1`, `B4`, …) refer to that document. Keep it open alongside this file.

**Update this file at the end of every work session**: tick boxes, record what was
verified, note anything that changed the plan. Sessions may be cleared between work
packages, so this file is the memory for *decisions and package state*.

Its companion is **`runs.md`**, the committed, append-only log of every training run —
what question it answered and what came back. Results live there, decisions live here.
This file drifted eleven commits behind the repo between 2026-09-20 and 2026-09-21
while `runs.md` stayed current; **reconcile against `git log` at the start of a session**,
not only at the end of one.

---

## Decisions taken (Part E, answered by Yoav 2026-09-20)

| # | Question | Decision |
|---|----------|----------|
| E1 | Three-way char-model comparison | **Rebuild it properly.** Controlled experiment, Claude runs the training here. |
| E2 | Shared model code | **Teaching notebooks generate the shared modules.** `nanochat.ipynb` builds the model cell by cell and those cells *write* `nanochat_model.py`; `bpe-tokenizer.ipynb` builds the merge loop and writes `bpe.py`. Downstream notebooks and scripts `import` them. See "Code reuse contract" below. |
| E3 | SFT task replacement | **TinyStoriesInstruct.** Note the repo name has **no hyphen**: `roneneldan/TinyStoriesInstruct`. See the verified download details in WP7. |
| E4 | GRPO clipping | **Add inner epochs (K>1)** so clipping engages; plot clipped-token fraction. |
| E5 | Biology connection | **Defer to a future revision.** Do not add biology content in this pass. |
| E6 | QK-norm (8.1) | **Measure, then fix and retrain** if confirmed. |
| — | Who runs the GPU jobs | **Claude, here**, on the 2× RTX A4000s. |
| — | Seeds | ~~**Single seed per run.**~~ **Reversed 2026-09-21: three seeds (42, 43, 44) with error bars**, for the character experiment at least. Run W4 in `runs.md` re-ran identical code and got rnn-3L 2.133 then 2.066 — GPU/XLA nondeterminism compounded over the run — so a single number could not be told apart from a real effect. The published sweep is 6 models × 3 seeds; the noise band it establishes (**seed sd ≈ 0.008, so a gap under ~0.015 bpc is noise**) is now the instrument every claim in the three notebooks is read against. The single-seed rule still stands for the nanochat work packages (WP6–WP8), where a run costs hours rather than minutes. |
| — | Checkpoint distribution | **Students do not get the `.pkl` files.** Deferred; do not add checkpoint URLs to `download_data.py` in this pass. Every notebook must therefore be runnable from scratch, or state plainly that it needs a checkpoint the student must produce. |
| — | Delivery date | **None set.** Size training runs for correctness, not for a deadline. |
| — | **Register: teaching, not research** | **Yoav 2026-09-24.** These are workshop notebooks. The main line demonstrates *how a mechanism works*; statistical care stays, but compressed to a sentence, not made the climax. WP4's Discussion was rebuilt on this basis (mechanism first: *how does each architecture compute "do cards i and j share a rank?"*). **Applies to WP5–WP10.** Noise bands, reproducibility and seed variance belong in `runs.md`, not in notebook prose. |
| — | **One story per notebook** | **Yoav 2026-09-24.** When a second, genuinely different lesson turns up inside a notebook, split it out rather than carry both. WP4 found a class-imbalance story inside a transformers notebook: the notebook keeps the *diagnosis* (it explains the transformer's own result) and the *fix* was handed to an FFN-only session in `yoavram/DataSciPy` — **issue #15**, filed with all numbers and code. Test to apply: does this lesson explain the notebook's own subject, or is it a different subject that happens to appear here? |
| — | **Ceilings before blame** | Established in WP4, generalisable. Before attributing a model's failure to its architecture, compute what the *task* permits — WP4's suit-blind ceiling (99.82% / 7-of-9) was derived from the data in three lines with no model, and predicted the trained Set Transformer's score exactly. Cheap, and it converts an unexplained blemish into the notebook's strongest claim. Worth asking in WP6–WP8 too. |

### Environment facts verified 2026-09-20
- 2× NVIDIA RTX A4000, 16 GB each — but **nothing in this repo is multi-GPU** (no `pmap`,
  no sharding; JAX takes device 0). Treat it as one GPU plus the ability to run a second
  job concurrently.
- **Driver 550.107.02**, so the CUDA pin must be reconciled toward **cuda12**.
  `index.ipynb`'s `jax[cuda13]` needs a ≥580 driver and will not run here (1.5).
- **There is no working Python environment.** The kernel every committed output came from
  (`~/miniforge3/envs/DataSciPy`) is gone, `.pixi/` has never been created, and system
  `python3` has no numpy. `pixi.lock` does cover linux-64 with `jax_cuda12_*-0.9.2-cp314`.
- Disk is at **94% (53 GB free)**. A pixi CUDA env (~8 GB) plus a second `train_tokens.npy`
  alongside the old one is uncomfortable.
- **`qwen3.5:9b` is a published Ollama tag — verified 2026-09-20** against
  `ollama.com/library/qwen3.5/tags`. Published sizes: 0.8b (1.0 GB), 2b (2.7 GB),
  4b (3.4 GB), 9b (6.6 GB, `latest`), 27b (17 GB), 35b (24 GB), 122b (81 GB). **1.6 and
  11.1 are closed.** Use **`qwen3.5:4b`** as the low-VRAM fallback.
- All checkpoints exist locally under `checkpoints/` (gitignored, so not on GitHub):
  `bpe_tokenizer.pkl`, `nanochat{,_sft,_grpo}_{checkpoint,best}.pkl`,
  `{rnn,gru,transformer}-jax-params-*.pkl`. `train_tokens.npy` (8.1 GB) is in the repo root.
- Stray duplicates in the repo root to clean up: `nanochat_best.pkl`, `nanochat_checkpoint.pkl`,
  `bpe_tokenizer.pkl` (symlink), `train_tokens.npy`.
- Checkpoint config (read from the pickle): `vocab_size=1024, d_model=512, n_heads=8,
  n_layers=8, d_ff=2048, head_dim=64, seq_len=256`, **26,223,104 params**.
- `train_tokens.npy` is `int64`, shape `(4132598, 257)` = **1.06 B tokens, 8.50 GB**.
  Vocab is 1024, so `uint16` would cut it to 2.1 GB — see the OOM note in WP-T.
- `data/` holds **only** the 22.5 MB valid split; the 2.23 GB train split is not on disk.
- Measured wall-clock on this hardware, from committed notebook `execution` timestamps:
  GRPO training cell **4821 s** (150 steps), whole GRPO notebook **1 h 42 m**; SFT training
  160 steps **28.5 s**; un-jitted single-token forward **≈55 ms**.

---

## Code reuse contract (E2)

Two different things get shared between notebooks, by two different mechanisms. Do not
confuse them.

**Data → pickle.** `checkpoints/bpe_tokenizer.pkl` holds merges and vocab.
`checkpoints/nanochat_checkpoint.pkl` holds config and params. Numbers only.

**Code → a module generated by the teaching notebook.** A pickle cannot carry `bpe_encode`,
`forward` or `attention_forward`, which is precisely why `nanochat-sft.ipynb` and
`nanochat-grpo.ipynb` today re-paste the whole model as one opaque cell, and why those
copies have drifted apart (`make_rotary` alias, inline `jnp.triu` mask, `load_checkpoint`
returning a 4-tuple instead of a dict).

So:

| Module | Generated by | Imported by |
|---|---|---|
| `bpe.py` | `bpe-tokenizer.ipynb` | `nanochat.ipynb`, `nanochat-chat.ipynb`, `minisweagent.ipynb` (11.6) |
| `nanochat_model.py` | `nanochat.ipynb` | `nanochat-sft.ipynb`, `nanochat-grpo.ipynb`, `nanochat-chat.ipynb` |

**`nanochat_chat.py` no longer exists** — Yoav's decision 2026-09-20: it became
`nanochat-chat.ipynb`, sitting between GRPO and the agent notebook. See WP-C.
**The character notebooks share no code at all.** Yoav's decision 2026-09-20: delete both
`charlm.py` and `char-model-comparison.py`. Each notebook owns its model *and* its harness.

What must be identical across the three is shared as **data, not code** — the same
pickle-as-interface pattern used for BPE and nanochat:

| Artifact | Written by | Read by |
|---|---|---|
| `checkpoints/charlm_split.npz` — encoded train/val token arrays + vocab | `RNN.ipynb` | `GRU.ipynb`, `text-transformer.ipynb` |
| `checkpoints/charlm_config.json` — step budget, batch size, context length, LR schedule, state policy, seed | `RNN.ipynb` | `GRU.ipynb`, `text-transformer.ipynb` |
| `checkpoints/charlm_result_{rnn,gru,transformer}.json` — bpc + run manifest | each notebook | every *later* character notebook |

Each notebook writes the sampler, the bpc helper and the evaluator itself. They are short,
and writing them is arguably better teaching than importing them.

**The comparison table is cumulative and lives at the end of each notebook, not on the
landing page.** Each notebook finishes by loading whatever result JSONs exist, checking the
manifests agree, and printing the table so far: `RNN.ipynb` shows the baselines plus RNN,
`GRU.ipynb` adds GRU, `text-transformer.ipynb` shows all three. The comparison is then a
result the student just produced rather than a claim they must take on faith before the
course starts, and it builds along the arc. If an earlier notebook has not been run, show
the rows that exist and say which are missing — never silently omit a model.

**Known trade-off, accepted:** the sampler and evaluator exist in triplicate and can drift.
The manifest check catches a mismatch *after* the runs, not before, so a mismatch costs a
re-run. Mitigate by making the manifest carry a **hash of the val token array** and an
explicit `state_policy` string — the two things most likely to differ silently are then the
two things checked hardest.

**Rules.**
- The teaching notebook builds each function cell by cell, with prose, exactly as now. Those
  same cells *emit* the module, so the file is by construction identical to what was taught.
  Never hand-edit the generated module.
- **The emission mechanism is not yet decided and must be before WP5 starts.** Two traps:
  plain `%%writefile` **does not execute the cell**, so the student reads code the kernel
  never defines; and an append helper is **not idempotent** — re-running or out-of-order
  execution duplicates or scrambles functions. A workable shape is per-cell `%%writefile -a`
  driven by one "regenerate module" cell that rewrites the file from a recorded ordered list,
  plus an `importlib.reload`. Somebody has to design and test this first.
- **`bpe.py` and `nanochat_model.py` are tracked in git**, so from WP5 onward every student
  who runs the notebook gets a dirty working tree. Decide: commit the generated files and
  accept that, or gitignore them and generate on first run.
- A generated module carries a header saying which notebook produced it and that edits belong
  in the notebook.
- Downstream notebooks `import` the module and load the pickle for params. They must not
  re-paste model or tokenizer code.
- `nanochat_model.py` exposes at minimum: `init_params`, `forward`, `attention_forward`,
  `precompute_rope`, `causal_mask`, `save_checkpoint`, `load_checkpoint`, `generate`.
- `load_checkpoint` returns a **dict**, everywhere, with no exceptions.
- Guard against a stale module. Byte-identical regeneration is the ideal but is not free
  (notebook-emitted files are not automatically stable), so start with the cheap version:
  the generated header records the source notebook, and a short check confirms the module
  imports and round-trips a checkpoint. Escalate only if drift actually appears.
  *(This rule and the export list below are authored, not review findings.)*

---

## Ground rules

- One work package per branch, one PR each. Finding IDs go in commit messages.
- Notebooks: `Read` + `NotebookEdit` only. Never parse `.ipynb` with bash Python.
- Cell indices in the review are 0-based positions in the *committed* notebook and shift on
  every insertion. Locate cells by the quoted source snippet, record the `id`, edit by `id`.
- **Never hand-edit an output cell.** Either re-run and commit real outputs, or strip the
  stale outputs and say so.
- Do not run the cells listed under "Do not run" in the review before the corresponding fix.
- **Source size must be watched *during* a package, not only at its start.** WP4 measured
  `sets.ipynb` comfortably under the limit, then crossed it an hour later by adding ~3k tokens
  of prose and one output table (29,577 executed, `Read` refused). Recovered by cutting printed
  training logs from up to 200 lines per model to 20 — evaluation still on a 100-point grid, so
  the loss curves were unchanged. Verbose training logs are the cheapest thing to cut and are
  pedagogically worthless.
- **A notebook over ~25k tokens of source cannot be edited by any tool.** `Read` refuses it,
  `NotebookEdit` needs that `Read`, `Edit` refuses `.ipynb`. Land prose and code edits
  *before* executing, and watch source size rather than output size — see WP-N and
  `CLAUDE.md`. The three character notebooks are 4.4 nbformat with **no cell ids**;
  `NotebookEdit` addresses them positionally as `cell-N`, and an `insert` shifts every
  later index.
- **`runs.md` is the experiment log and is committed; this file is the plan and is not.**
  Results go in `runs.md`, append-only, superseded numbers kept with a note. Decisions,
  package state and reversals go here. Do not duplicate a results table across both.

---

## Compute budget (estimated on one A4000; measured figures marked)

**The character-model jobs outgrew this machine and moved to the TAU Slurm cluster**
(2026-09-21). Seven array jobs, C1–C7, are logged in `runs.md`; the plan below never
scoped the depth (C5, C7) or context-length (C6) sweeps at all. `cluster/` holds the
submission scripts and `cluster/README.md` the runbook; the `cluster-agent` skill is the
procedure. An A4000 is 1.7–1.9× slower than the cluster's A6000 across all six models
(W6), so the estimates below hold as A4000 figures and should be divided by ~1.8 for the
cluster.

| Job | Estimate |
|---|---|
| ~~WP3, three char models, 20 k steps~~ | Superseded. **Measured**: 6 models × 3 seeds × 30 k steps = 18 tasks, **40 min of A6000 wall-clock spread over an array job** (C4). Per model: transformer 1.2 min, rnn 4.0, gru 5.8, transformer-3L 2.0, rnn-3L 11.2, gru-3L 15.7. The same six on one A4000 take 57 + 78 + 18 min as three serial notebooks (W6). |
| Depth sweep, depths 2/4/6 × 3 archs × 3 seeds (C5) | **27 tasks**; the expensive tail is gru-6L at 30.9 min |
| Context sweep, 256/512 × 3 archs × 3 seeds (C6) | **18 tasks**; the expensive tail is gru-3L at ctx 512, 61.8 min |
| Deep-transformer extension, 8L/12L (C7) | **12 tasks**; transformer-12L 4.7 min against gru-8L's 40.3 |
| Re-executing the three character notebooks for committed outputs | **57 + 78 + 18 min** on the A4000, *measured* (W6) — pay this again after any prose edit |
| WP6 QK-norm A/B, 2 k steps × 2 | **~30 min** |
| WP6 nanochat retrain, 50 k steps × 32 × 256 = 410 M tokens | **3.5–7 h** |
| WP-T retokenise, if the train split is used | 2.23 GB download + **1–3 h** pure-Python encode |
| WP7 SFT, 10 k steps × 32 | **< 1 h**; real costs are BPE-encoding the dataset (572 s *measured*) and un-jitted `generate` (407 s for 200 tokens *measured*) |
| WP8 GRPO as-is | **1 h 42 m** *measured* |
| WP8 GRPO with the plan's additions, sampling unfixed | **8–12 h** |
| WP8 GRPO with `sample_group` batched and jitted | roughly 20× better; re-measure |

For reference, the original nanochat run early-stopped at 23,500 steps (~192 M tokens,
**0.18 epoch**) for val 1.069 nats/token; removing early stopping per B5 roughly doubles it.

---

## Work packages

### WP0 — Repo hygiene  ✅ done 2026-09-20 (branch `wp0-repo-hygiene`, commit `713d089`)
No behavioural change. Merge first so later diffs are readable.

- [x] **Environment created and verified 2026-09-20.** `pixi` lives at `~/.pixi/bin/pixi`
      (v0.65.0); `pixi install` succeeded on linux-64 giving **jax 0.9.2 with the GPU
      backend**, both A4000s visible, and three models trained on it end to end. Disk went
      94% → 95% (47 GB free). Note the committed notebook outputs were produced on **CPU**,
      so GPU reruns will not match them bit for bit (see WP4).
- [x] **`bpe.py` deliberately skipped.** Its `DEFAULT_TOKENIZER_PATH` is already
      `checkpoints/bpe_tokenizer.pkl`, so nothing needed doing. Under the code reuse contract
      it becomes a generated file that WP5 overwrites; do not hand-edit it.
- [x] `nanochat_chat.py` adopted: its `CHECKPOINTS` list now names `checkpoints/`. Its
      `top_k`/`top_p` sampling still has to be reconciled with `generate` in **WP6**, and its
      `load_checkpoint` still returns a 4-tuple — **WP6 owns both**
- [x] `.gitignore` rewritten. The blanket `*.txt` + allowlist is gone, replaced by
      `*.pkl`/`*.npy`/`*.npz`, `data/TinyStories*.txt`, `data/poker-hand-*.data` and
      `data/cache/`. WP4's `.data` mirrors are now ignored and WP9's cached `.txt` fallback
      is no longer silently ignored. `checkpoints/charlm_result_*.json` (WP1) is **not**
      ignored, as WP1 requires

- [x] Delete `set-transformer.ipynb` (3.1); only `plan.md` / `nanochat-review.md` /
      the handoff ever mentioned it
- [x] `README.md`: TinyStories valid split is 22.5 MB, not "~10 MB" (1.2); poker data listed
- [x] All save/load paths name `checkpoints/` (1.3, B4, 4.1, 7.3). RNN/GRU/text-transformer
      were saving to `data/` while loading from `checkpoints/`; bpe-tokenizer saved to the
      repo root. No helper was added — each notebook owns its own path, per the WP1 contract
- [x] `nanochat.ipynb` writes `checkpoints/train_tokens.npy` (B4)
- [x] **Moved** root `train_tokens.npy` (8.5 GB) into `checkpoints/`; removed the
      md5-verified duplicate root `nanochat_best.pkl` / `nanochat_checkpoint.pkl` and the
      `bpe_tokenizer.pkl` symlink
- [x] `checkpoints/.gitkeep` is now tracked, so a fresh clone has the directory the
      notebooks write into *(authored, not a review finding)*
- [x] Setup reconciled (1.4): `README.md` is canonical, `index.ipynb` links to it
- [x] CUDA pin reconciled **toward cuda12** (1.5). `pixi.toml` was already cuda12; it was
      `index.ipynb` that advertised `jax[cuda13]`
- [x] `jnp` everywhere (B3); RNN/GRU/text-transformer no longer bind `np` to JAX
- [x] ~~`download_data.py`: add checkpoint URLs (B6)~~ — **dropped this pass**. But the
      2.23 GB train split is now opt-in behind `--train` (A3), so a student's first run no
      longer pulls it. `RESUME` (8.8) belongs to WP6
- [x] Typos and dead code: "Intoduction" (1.8), `nanochst` (9.2), `pip instal` (11.7),
      duplicate line in RNN cell 6 (4.10), stale timing comment (4.11), unused imports (4.13),
      duplicate `import os` (6.7), `ipywidgets` added to `pixi.toml` for tqdm (7.9)
- [x] Broken/wrong references: arXiv:1607.06450 not 06515 (4.4), `FFN.ipynb` link (4.5),
      GRPO → DeepSeekMath arXiv:2402.03300 (10.7)

**Acceptance — all met.** `set-transformer.ipynb` gone and unreferenced; no `.pkl`/`.npy`
path outside `checkpoints/` remains in any source cell; no notebook binds `np` to
`jax.numpy`; README, `index.ipynb` and `pixi.toml` agree on setup and on the valid split.
Smoke test passed on the pixi env: jax 0.9.2 **gpu**, both A4000s visible, ipywidgets 8.1.9,
`bpe_load()` round-trips through the `checkpoints/` default, `nanochat_chat.find_checkpoint()`
resolves.

**⚠ Outputs stripped from `RNN.ipynb`, `GRU.ipynb`, `text-transformer.ipynb`.** Editing the
source cells cleared most outputs as a side effect of `NotebookEdit`, and a half-stripped
notebook is worse than a clean one, so the remaining 4–6 per notebook were cleared with
`jupyter nbconvert --ClearOutputPreprocessor.enabled=True`. Permitted by the ground rule
("strip the stale outputs and say so"), and cheap here because **WP2 rewrites all three and
WP3 re-runs them**. Nothing else lost outputs: the one-line fixes to `nanochat.ipynb`,
`nanochat-sft.ipynb`, `nanochat-grpo.ipynb`, `bpe-tokenizer.ipynb` and `minisweagent.ipynb`
were literal in-place substitutions, so those notebooks keep every committed output.
**WP2 must therefore re-run RNN/GRU/text-transformer and commit real outputs** — there is no
longer a fallback of "the old outputs are still there".

---

### WP1 — the shared experiment, as data  ✅ done (branch `wp1-shared-experiment`)
Reviewed by subagent 2026-09-20; two acceptance items had been ticked in error. **All three
S1 blockers are now fixed and verified in commit `3796b89`** — which also pulls **A2 and the
rest of A3 forward from WP2**, because a model cannot be scored through `evaluate_bpc`
one window at a time. Remaining S2/S3 findings are listed at the end and are **not** done.

**The controlled comparison now exists**, all five rows under one protocol, one seed, one
budget, with real committed outputs in all three notebooks:

| model | bpc | params |
|---|---|---|
| unigram | 4.770 | — |
| bigram | 3.581 | — |
| rnn | 2.078 | 331,331 |
| gru | 2.051 | 925,251 |
| transformer | **1.952** | 1,019,843 |

Ordering is monotone in parameter count as well as in architecture, so **this does not
isolate architecture from capacity** — say so wherever the table is discussed. RNN takes
~3 min for the full 10,000 steps at 19 ms/step.
Enables A1–A4. **No shared `.py` file** — `charlm.py` and `char-model-comparison.py` are both
deleted. `RNN.ipynb` builds the harness cell by cell and writes the split and config
artifacts; the other two notebooks read those artifacts and write their own harness code.

- [x] Loader for `data/shakespear3.txt`, integer encoding, **no one-hot** (A3). Two
      streaming passes + a codepoint lookup table: **15.2 MB peak** for Shakespeare,
      36.3 MB for TinyStories. The old `X = onehot_encode(text)` was a 1.2 GB float array
- [x] One fixed train/val split, contiguous (last 10%), shared by all three notebooks
- [x] Minibatch sampler, leading batch dim, `(64, 128)` int32
- [x] Bits-per-character helper (A4), plus `windows()` and `evaluate_bpc()`
- [x] Unigram and bigram baselines **through the same `evaluate_bpc` entry point** as the
      neural models. **Correction (review):** the earlier claim that this "fixed the A1
      confound in miniature" was overstated. Measured, windowed bigram is **3.580837** and
      streaming bigram is **3.580836** — identical to six figures, because a bigram's
      context is one character and every window's first position has a real predecessor.
      The value is code-path hygiene, not a corrected number. Say that, not the stronger claim
- [x] `RNN.ipynb` writes `checkpoints/charlm_split.npz` and `checkpoints/charlm_config.json`;
      GRU and text-transformer **load** both and verify the val hash matches the config
- [ ] **NOT DONE — state policy is declared but not implemented (S1).** `charlm_config.json`
      says `state_policy: 'none'` and `RNN.ipynb` cell-6 says "every window starts cold",
      but the RNN and GRU training loops thread `h` out of one step and into the next
      (`params, h, opt_state, loss = update_params(..., h)`), resetting only at corpus
      wraparound. The transformer does not. **This is review confound A1, still live, now
      with a config file asserting it is gone** — worse than before, because the claim is
      now machine-readable. Fix: reset `h` to zeros every step, or rename the policy honestly
- [ ] **NOT DONE — no neural model is ever scored (S1).** `write_result` is *defined* in all
      three notebooks and *called* only for `unigram` and `bigram`, in `RNN.ipynb`. Nothing
      calls `evaluate_bpc` on trained parameters, and `n_params` is never computed. So
      `charlm_result_{rnn,gru,transformer}.json` can never exist and `results_table()`
      permanently prints `not run` for all three. Each notebook needs a post-training cell
      building a `logprob_fn` adapter — note RNN/GRU `step` returns **probabilities** (needs
      `jnp.log`) while the transformer returns **logits** (needs `log_softmax`), and the
      transformer's entry point takes one-hot — then `write_result(<model>, bpc, n_params=…)`
- [x] Each notebook ends with the same cumulative `results_table()`: reads whatever result
      files exist, **raises** if two manifests disagree on val_hash / state_policy /
      context_length / steps / batch_size / seed, prints `not run` for missing models
- [x] Deleted `char-model-comparison.py`
- [x] `.gitignore`: `*.npz` ignored (added in WP0) so `charlm_split.npz` stays out of git;
      `charlm_result_*.json` and `tinystories_baselines.json` are **not** ignored and are
      committed, so a reader who has run nothing still sees the baseline rows
- [x] **Baselines on the TinyStories val split as well**, written to
      `checkpoints/tinystories_baselines.json` for WP6

**Measured baselines (committed):**

| corpus | V | unigram | bigram | uniform = log2(V) |
|---|---|---|---|---|
| Shakespeare | 67 | **4.770** bpc | **3.581** bpc | 6.066 |
| TinyStories | 91 | **4.446** bpc | **3.292** bpc | 6.508 |

For scale: the old char models reported ~1.2–1.4 nats/char ≈ 1.7–2.0 bpc, so a trained
model roughly halves the bigram baseline. The evaluator is checked against a uniform model,
which must score exactly log2(V) — it does, to floating point.

**Chosen config** (revisable by WP2/WP3): context 128, batch 64, 10,000 steps = 82 M
characters ≈ 19.9 epochs, warmup 200 → peak LR 3e-3 cosine, grad clip 1.0,
`state_policy='none'`, seed 42.

**Acceptance — partially met.** Met: baselines run in ~2 s on CPU and report bpc on both
corpora; peak load memory 15.2 MB (Shakespeare), well under 50 MB; no `.py` file added —
both `charlm.py` and `char-model-comparison.py` are gone; GRU and text-transformer raise
`FileNotFoundError` naming `RNN.ipynb` when the artifacts are missing, and `ValueError` if
split and config disagree. **Not met:** the result-JSON item and the state-policy item above.

### Review findings, 2026-09-20 (subagent, independently spot-checked)

**S1 — blockers — ✅ ALL FIXED in `3796b89`**
1. ~~No neural model is ever evaluated or written.~~ Each notebook now scores its trained
   model through `evaluate_bpc` and writes its row with bpc, `n_params` and the manifest.
2. ~~`charlm_config.json` documents an experiment the code does not run.~~ Training is now
   genuinely `batch_size × context_length` per step, via `jax.vmap` over a batch of windows
   and `jax.lax.scan` over time. This is **A2, pulled forward from WP2**.
3. ~~State policy declared but not implemented.~~ `h0` is created inside `batched_logits` on
   every call, so the cold start is structural; `NLL` no longer carries state and `has_aux`
   is gone. **A1 is now actually closed, not just asserted.**

Additionally fixed while here: **S2-6** (RNN/GRU used unstable `log(softmax)`; all three now
use `log_softmax`), **S2-5** (checkpoints namespaced to `checkpoints/charlm/<model>-<step>.pkl`
— this was not hypothetical: the first fixed RNN run reported 2.415 bpc because the load cell
picked up the legacy 4.8M-step file and scored *that*; namespaced it reports 2.078), the GRU
multi-layer indirection (**5.3**), train/val curves plotted at a fixed budget with no early
stopping (**B5**), the transformer sampler's per-length recompile, and `index.ipynb`'s stale
uncontrolled table (**1.1**).

**S2/S3 — still open, for WP2/WP3.** Not blockers, but none of the following is done:
`split_corpus(tokens, 0.0)` returns an empty train split; `evaluate_bpc` divides by zero if
the token array is shorter than one window; `results_table()` still raises on one stale file
and prints zero rows instead of showing the good rows and naming the outlier; a UTF-8 BOM
would enter the vocabulary (`encoding='utf-8-sig'`); the uniform-model assertion is
`y`-independent and so blind to a `windows()` off-by-one (a windowed-vs-streaming bigram
check would be alignment-sensitive); the manifest still hashes the split but not the scoring
(finding 4 below); 37 legacy `*-jax-params-*.pkl` files remain in `checkpoints/`, now
unreachable but stale; `text-transformer.ipynb` still says attention is permutation
"invariant" (6.1); `encode()` in `RNN.ipynb` is unused. Overfitting at ~20 epochs with no
dropout or weight decay is now visible in the committed train/val curves — check them before
WP3 treats the budget as settled.

**S2 detail**
4. **The manifest check misses the drift it was introduced to catch.** It hashes the *split*,
   never the *scoring*. Undetectable: a changed `windows()`/`evaluate_bpc()`, scoring
   `train_tokens` by mistake (the val hash is copied from the data cell, not recomputed by
   the evaluator), or a recurrent evaluator carrying `h` while the string stays `'none'`
   (i.e. finding 3). Fix: have `evaluate_bpc` recompute `array_hash(tokens)` into the record.
5. **The legacy 4,800,000-step checkpoints now win the numeric sort** that `efc767f`
   introduced — 4800000 > 10000 — and load without error (shapes are compatible), so after
   WP3 the sampling cells would show text from the *pre-WP1* model. Namespace new runs under
   `checkpoints/charlm/` or delete the 15 legacy files before WP3.
6. **RNN/GRU use `jnp.log(softmax(...))`; the transformer uses `log_softmax`.** In a
   comparison whose premise is identical treatment, two of three use the unstable form. The
   30× LR raise makes it live: measured RNN loss spike to **13.4 nats** at step 550.
7. **LR 3e-3 is sound at batch 64, useless at batch 1.** Measured (400 steps, B=64):
   RNN 2.088 / GRU 1.719 / transformer 1.753 nats, all beating lr 1e-3 — so the config is
   fine *once WP2 lands*. At B=1 (what the code does today) the RNN reaches ~4.71 bpc,
   **worse than the bigram baseline (3.581)**. The table would show the flagship
   architecture losing to add-one counts.
8. `split_corpus(tokens, 0.0)` returns an **empty train split** (`tokens[:-0]` is `tokens[:0]`).
9. `results_table()` raises on one stale file and prints **zero** rows, contradicting the
   plan's "show the rows that exist and say which are missing". Its reference manifest is
   also just the alphabetically-first file.

**S3** — uniform assertion is weaker than advertised (y-independent, so blind to a
`windows()` off-by-one); `while batch <= max_batches` runs 10,001 steps against a manifest
saying 10,000; a UTF-8 BOM would enter the vocabulary (`encoding='utf-8-sig'`);
`evaluate_bpc` divides by zero if `len(tokens)-1 < context_length`; `update_params` closes
over a module-global `optimizer` rebound after jit; `index.ipynb` still asserts the old
uncontrolled loss table; dead code (`matplotlib`/`losses` never plotted, `encode()` unused,
`sample_batch` unused in GRU and text-transformer).

**Confirmed correct by the review:** `windows()` tiling (3,572 windows, 457,216 of 457,333
val characters scored, no double-counting, 116-char tail dropped); the bigram-at-window-start
claim; all four committed baselines reproduce byte-identically; Laplace α is immaterial
(α ∈ {1e-3, 0.1, 1} spans 0.0008 bpc); manifest *structure* is identical across the three
notebooks; `load_corpus` is correct including non-BMP characters; the missing-artifact and
hash-mismatch guards; `charlm.py` and `char-model-comparison.py` fully gone.

**Also fixed here, and not in any package:** `RNN.ipynb` and `GRU.ipynb` declared kernel
`pixi-default` and `text-transformer.ipynb` declared `conda-env-scipy-py`. **Neither kernel
exists**; only `python3` does. Every one of these three notebooks failed to start for a
student, and blocked `nbconvert --execute` here. All notebooks now declare `python3`.

**⚠ Still unexecuted.** All three notebooks remain output-free. Running them now would be
misleading: the training loop is still one window per step, so 10,000 steps sees 0.3 of an
epoch and produces garbage. WP2 adds real batching and **WP3 runs all three and commits the
outputs**. Correctness was verified on short-budget copies (`max_batches=100`), which
executed end to end: RNN and GRU clean; text-transformer clean once its sampler calls were
stubbed out. The sampler is slow for a pre-existing reason — un-jitted, it recompiles once
per window length, the same shape-instability bug fixed in WP-C, and worse now that the
context is 128 rather than 50. A full run did not finish in 25 minutes.
**WP2 must port WP-C's fixed-shape generation to `text-transformer.ipynb`**; until then its
sampling cells are impractically slow.

**Latent bug found and fixed (commit `efc767f`), not in the review.** All three notebooks
loaded the newest checkpoint with `sorted(glob.glob(...))[-1]`, which sorts paths as
strings. The old budget hid it — every step number was `0` or seven digits, so lexicographic
order happened to match numeric order. **WP1's 10,000-step budget breaks that**: 0, 1000,
… 10000 sorts with `-9000` after `-10000`. Now keyed on the parsed step number.

**Housekeeping note for WP3:** short-budget smoke tests write `*-jax-params-<small>.pkl`
into `checkpoints/`, which then shadow the real runs. 42 such files were deleted. The
original `*-jax-params-0.pkl` (random init) files were overwritten in the process and are
gone; they are regenerated by any run and nothing depends on them. **Use a scratch
checkpoint directory for smoke tests.**

---

### WP2 — Rewrite `RNN.ipynb`, `GRU.ipynb`, `text-transformer.ipynb` on WP1  ✅ done 2026-09-21
Does A2, A3, 4.1–4.14, 5.1–5.5, 6.1–6.4, 6.6, 6.8 — **minus the rows already closed in WP0**
(4.1, 4.4, 4.5, 4.10, 4.11, 4.13, 6.7).

**This package was never worked as a package.** It was absorbed into the character-model
campaign of 2026-09-20/21 (commits `3796b89` … `4abebda`), which pulled its batching work
forward into WP1 and then rewrote all three notebooks against the cluster results. The
ticks below were verified against the committed notebooks on 2026-09-21.

**Closed by Yoav 2026-09-21 with four rows outstanding** — they are prose-only, none
changes a number or a claim, and `text-transformer.ipynb` is past the edit limit anyway.
They are struck through below rather than deleted, so a later pass can pick them up if it
is already in the file for another reason. The GRU layer-norm question is the only one with
any substance left in it and survives as an open item at the foot of this file.

- [x] Real minibatching: `jax.vmap` over a batch of windows + `jax.lax.scan` over time for
      RNN/GRU; leading batch dim for the transformer (A2) — **landed early, in `3796b89`
      under WP1**, because scoring a model through `evaluate_bpc` needs batched forwards
- [x] Drop the one-hot pipeline; `W[:, i] == W @ onehot(i)` gets one markdown cell (A3, 6.3)
- [x] Fixed step budget, honest wall-clock, **no early stopping** (B5, 4.8, 4.9) — 30,000
      steps, `no early stopping` stated in all three, minutes a table column
- [x] Train *and* val curves plotted; losses in bpc next to the WP1 baselines
- [~] ~~Name truncated BPTT in RNN and GRU (4.6)~~ — **dropped.** The string "BPTT" appears
      in none of the three notebooks. The A1 contrast it was meant to set up is now carried
      structurally by `state_policy='none'`, which is stated in every table caption
- [x] RNN: "22,308 updates" recomputed from `data_size` (4.2); `vocab_size` used
      symbolically rather than "62 characters" (4.3); prompt-conditioned `sample` (4.14)
- [~] ~~RNN: justify or move the post-`tanh` LN (4.7); replace ResearchGate hotlinks
      (4.12)~~ — **dropped.** `researchgate` still appears twice in `RNN.ipynb` and once in
      `GRU.ipynb`; cheap to fix whenever those notebooks are next open
- [~] ~~GRU: move LN into the pre-activations~~ (5.2, downgraded to **S2** — see
      Corrections) — **deferred, not dropped.** `Ba et al` is now cited, so the nonstandard
      placement is at least named, but the ablation the plan asked for *before* writing
      prose was never run. Survives as an open item at the foot of this file
- [x] GRU: the multi-layer indirection is now used, not decorative (5.3) — real stacked
      implementations written in `57b55df`, and depth is the measurement the whole campaign
      turns on
- [x] text-transformer: permutation-**equivariant**, not invariant (6.1); stateless-window
      vs carried-state trade-off stated (6.2) — `state_policy='none'` is structural and
      named in every table caption; LR equalised across all three at `peak_lr 3e-3` (6.4)
- [~] ~~text-transformer: add nanoGPT + a pre-norm/post-norm pointer (6.8)~~ — **half
      done, closed.** Pre-norm is discussed; nanoGPT is not cited
- [x] Re-run all three; commit real outputs — W6, `nbconvert --execute --inplace`, zero cell
      errors, 57 + 78 + 18 min on the A4000

**Deferred:** KV caching (6.5) — an optional section, park it unless time allows.

**⚠ Editing constraint discovered the hard way.** `text-transformer.ipynb` is 25.5k tokens
with every output stripped, past `Read`'s 25k limit on its prose alone, so `NotebookEdit`
cannot touch it and `Edit` refuses `.ipynb` outright — which is part of why the rows above
were closed rather than finished. See WP-N, `CLAUDE.md` and runs.md W6.

---

### WP3 — The controlled comparison  ✅ done 2026-09-21, and overtaken
Does A1. **Delivered far beyond what this section specifies** — the campaign in `runs.md`
answered A1 and then kept going into depth and context sweeps that were never planned. Read
`runs.md` for the authoritative account; this section is kept for the finding-ID trail.

Two things in the original specification were **reversed by measurement** and the text
below has been corrected rather than deleted:

- ~~Single seed, stated in the script and the table caption (no multi-seed averaging)~~
  → **three seeds, 42/43/44, with error bars.** Reversed on the evidence of W4; see the
  Seeds row in the decisions table.
- ~~20 k steps, ~1 M parameters~~ → **200,000 parameters, 30,000 steps.** W1 found 500k
  overfit a 4.6 MB corpus (gru peaked at 18k steps, final ≫ best); W2 found 10k steps
  left every model still improving. Retargeted in `1fa42d0`.

- [x] Identical split, token budget, LR schedule across all three models — pinned in
      `checkpoints/charlm_config.json`, widths *solved* per depth so depth is bought out of
      width
- [x] Identical evaluation: same context length, same state policy for all three,
      named in the table caption — `state_policy='none'`, context 128, batch 64. **The
      decisive confound A1 is closed structurally**, not by convention: `h0` is built inside
      `batched_logits`, so no state can be carried
- [x] Report bpc, with the WP1 unigram/bigram baselines in the same table
- [x] **Each notebook collates what exists**: `results_table()` reads the
      `charlm_result_*.json` files present and hash-checks the manifests; `depth_table()`
      (added `40bf38d`) renders the depth curve from the 39 per-seed JSONs under
      `checkpoints/charlm-depth/`. No central collator
- [x] **Delete the comparison table from `index.ipynb`** (1.1)
- [x] Independently reproduced **three times** — W5 (in-notebook, A4000), C4 (cluster
      A6000, a second independent implementation in `cluster/charlm_run.py`), W6 (A4000
      re-execution). Every pairwise gap is inside the 0.015 noise band

**The published numbers (C4, 6 models × 3 seeds × 30k steps):**

| model | params | mean bpc | sd |
|---|---|---|---|
| unigram | — | 4.770 | — |
| bigram | — | 3.581 | — |
| rnn | 200,267 | 2.0966 | 0.0072 |
| gru | 200,141 | **2.0607** | 0.0137 |
| transformer | 204,907 | 2.1775 | 0.0064 |
| rnn-3L | 200,531 | 2.0844 | 0.0058 |
| gru-3L | 201,441 | **2.0421** | 0.0075 |
| transformer-3L | 206,635 | 2.0443 | 0.0058 |

**What the experiment actually concluded, and it is not what the review expected.** At a
fixed 200k budget the **GRU wins on bits per character** — at 1 layer the transformer is
the *worst* of the six, and at best depth gru-2L (2.028) and transformer-6L (2.041) are
tied inside the noise band. The transformer claim had to be rebuilt on two other axes:

1. **Cost (the main argument).** transformer-3L reaches gru-3L's quality **8× faster**
   (2.0 min vs 15.7); 12 transformer blocks cost 4.7 min against 8 GRU layers' 40.3.
   This is Vaswani Table 1 — `O(1)` sequential operations vs `O(n)` — measured.
2. **Depth tolerance (supporting, and confounded).** rnn-8L reaches 4.462 bpc, *worse than
   the bigram baseline*; gru-8L 3.369 with seeds spread over 2.2 bpc; the transformer moves
   only 2.041 → 2.056 between depth 6 and 12. **Confound stated in the notebook:** our
   recurrent stacks have no residual connections and the transformer does, so this partly
   measures residual streams. The cost argument survives the confound; this one does not.

**Two prose claims were written and later falsified by our own data** — recorded because
the pattern repeated: both were curves read past their last measured point.
"The transformer improves monotonically" died on depth 3 → 4 (+0.024, 3σ), and "sits on a
curve still descending" died on C7 (the optimum is depth 6; 8 and 12 are worse). Corrected
in `40bf38d` and, for the C7 half, **not yet** — see the open item below.

---

### WP-N — Get the character notebooks back under the edit limit  ⬜ not started
Not in the review; created 2026-09-21 by a tooling constraint that now blocks WP2.

`text-transformer.ipynb` is **25.5k tokens of source with every output stripped**, and
~28.7k once executed. `Read` refuses over 25k and has no working offset for `.ipynb`;
`NotebookEdit` requires a `Read` in the same session; `Edit` refuses `.ipynb`. So there is
**no tool path to an edit** on that notebook. The one prose fix that had to land anyway
(`a97e6d2`) went through the `nbformat` API by hand and cost a full 18-minute re-execution.

Shrinking the figure does not help — `dpi=72` cut the PNG from 44 KB to 28 KB and the file
from 129 KB to 113 KB while the token count went 28,756 → **29,043**. Bytes and tokens are
not proportional across base64, and the notebook is over the limit on prose alone.

- [ ] Factor `results_table()` / `depth_table()` into a module — ~120 lines triplicated
      across the three notebooks, **plumbing rather than pedagogy**. Estimated to take
      `text-transformer.ipynb` from 25.5k to ~24k stripped tokens, which restores the
      clear → edit → re-execute route
- [ ] Leave the deliberately duplicated teaching helpers inline: `sample_batch`,
      `evaluate_bpc`, `repeat_seeds`. Each notebook keeps its own harness — Yoav's
      decision, 2026-09-20; this package must not quietly undo it
- [ ] Re-execute all three and commit outputs (57 + 78 + 18 min on the A4000)
- [ ] Only then: the C7 conclusion fix (delegated to its own issue), and any
      struck-through WP2 prose rows worth picking up while the file is already open

**Sequencing rule this package exists to enforce: land every prose and code edit *before*
executing.** Clearing outputs afterwards does not necessarily buy back enough room — here
it did not.

---

### WP4 — `sets.ipynb`  ✅ done 2026-09-24 (branch `sets-revision`, `c827495` → `f753ff4`)
Does 2.1–2.10. **No retraining** — Yoav's decision 2026-09-20: keep the training exactly as it
is and change the metric. Honoured: no model, budget or hyperparameter changed. The notebook
*was* re-executed, because no sets checkpoints exist and the ground rules forbid hand-editing
outputs — re-execution is how real outputs get made, not a retraining decision reversed.

**Committed results** (CPU, jax 0.9.2; full numbers live in the notebook's outputs and the
cross-run picture in `runs.md` S1 — not duplicated here):

| | Params | Test | Val macro |
|---|---|---|---|
| Flattened FFN | 20,150 | 99.0% | 68.2% |
| Deep Sets | 20,152 | 77.4% | 27.3% |
| Set Transformer | 20,010 | 99.8% | **77.8% = exactly 7/9** |
| Majority baseline | — | 50.1% | 11.1% |

**Yoav's steer, 2026-09-21: this is pedagogical material, not research.** The Discussion had
been drifting toward noise bands and reproducibility — the register of `runs.md` — and was
rebuilt around the mechanism instead: *how does each architecture compute "do cards i and j
share a rank?"* FFN learns it C(5,2)=10 times, once per position pair; Deep Sets never places
two cards side by side, so it must smuggle the coincidence through a sum; attention computes it
directly with shared weights, so it learns it once. Statistical caveats kept, but cut to a
sentence each. **Apply this steer to WP5–WP10 as well.**

- [x] **2.2 the central change.** Per-class recall, macro average (= balanced accuracy),
      majority baseline row, and prose on micro vs macro. Royal flush has no validation
      examples so the macro average is over nine classes — which is why the majority baseline
      is 1/9, not 1/10
- [x] **2.3 per-hand permutations.** **Result was not what the review expected**: the stronger
      test *agrees* with the weak one (1.00% vs 1.10%), so it is reported as agreement, not
      discovery. A planned exercise built on the opposite expectation was replaced — see below
- [x] **2.1 step-budget asymmetry stated**, not equalised; exercise 1 asks students to equalise
- [x] 2.5 Zaheer hedged — trimmed under the pedagogical steer to the one caveat that pays off:
      the theorem says a solution *exists*, not that training finds it
- [x] **2.6 — the review describes this backwards.** It says cell 25 "has `\\` line breaks that
      render wrong"; the defect is that three equations shared one `$$…$$` with **no** `\\`.
      Fixed with `aligned`
- [x] 2.7 train/val/test all on full splits (train had been a 10,000-row subset)
- [x] **2.8 reversed `split` operands — not cosmetic.** It swapped which key continued the
      chain, so fixing it shifted the key stream for Deep Sets and the Set Transformer. Every
      number in the notebook moved; this is why prose numbers had to come from the notebook's
      own execution and not from the side runner
- [x] 2.9 six exercises
- [x] ~~2.4 train on 5 / eval on 7~~ dropped 2026-09-20 — **but the "variable-size input ✓"
      claim it left asserted-and-unshown is now recovered honestly by exercise 2**: five-card
      data cannot demonstrate *larger* sets, but dropping a card demonstrates smaller ones —
      the FFN raises, the other two run
- [x] 2.10 closed 2026-09-20 (UCI URL resolves)
- [x] Not from the review: `os.makedirs` for the download dir (cell failed on a fresh clone);
      the permutation test no longer shadows the dataset-shuffle `perm`; `rho_*` indentation;
      wall time captured into variables so the table can report it

**Three things worth carrying forward.**

1. **`sets.ipynb` crossed the 25k-token edit limit mid-package** (29,577 executed) and `Read`
   refused it — the WP-N trap, reached *from under the limit* by adding ~3k tokens of prose
   plus one output table. I had told Yoav this notebook was safely clear of the wall; that was
   true when measured and false an hour later. Recovered by cutting printed training logs from
   up to 200 lines per model to 20, evaluation still on a 100-point grid so the curves are
   unchanged. **Watch source growth during a package, not only at its start.**
2. **A claim was corrected before commit, not after.** A draft said the FFN "fades as hands get
   rarer". The measured column is not monotone — 100% on straights (support 394) against 48.7%
   on full houses (150) and 75.0% on four of a kind (20). Same class of error as C5 and C7.
4. **The flush explanation committed in `109d6e4` was wrong, and was replaced in `f753ff4`.**
   It blamed rarity; the notebook's own support column refutes that (full house 150 and four of
   a kind 20 both score 100%, against flush's 180 at 0%). The real cause is that **rank alone
   separates 7 of the 10 classes**, leaving `Nothing|Flush` and `Straight|Straight flush` — so
   the suit channel is worth ~0.2% of accuracy and the suit-blind ceiling is 99.82% / 7-of-9,
   which is *exactly* where the Set Transformer landed. I reached for the standard
   class-imbalance story without checking it against a table I had already produced.
5. **Scope decision, Yoav 2026-09-24: the diagnosis stays, the fix leaves.** `sets.ipynb` is a
   transformers notebook, so it carries the ceiling analysis plus two diagnostics (embedding
   norms; a linear probe on frozen features recovering 66.1% flush recall above the ceiling)
   and stops. Resampling, class weighting, thresholds and metric choice go to an FFN-only
   session in `yoavram/DataSciPy` — **issue #15**, filed 2026-09-24 with every number, the
   reusable ceiling code, and the caveats. Keeps this notebook to one story and ~35 min of
   execution. Numbers live in `runs.md` S2.

6. **The Deep Sets non-reproducibility has a better explanation than the plan's.** The plan
   blamed CPU-vs-GPU; two further CPU runs (85.1%, 77.6%) refute that. Yoav's reading — one
   learning rate shared by three models, and Deep Sets alone sums five vectors before `rho`
   sees them — fits the plateau-escape times far better. Untested; it is exercise 6.

Committed results (CPU), for reference:

| Model | Params | Train | Val | Test | Wall time |
|---|---|---|---|---|---|
| Flattened FFN | 20,150 | 99.5% | 99.6% | **99.5%** | 37.9 s |
| Deep Sets | 20,152 | 92.4% | 92.1% | **92.0%** | 4 min 24 s |
| Set Transformer | 20,010 | 99.8% | 99.8% | **99.8%** | 4 min 10 s |

Parameter counts are **already equalised** at ~20k, so 2.1 is about unequal *steps*
(Deep Sets 200k vs 50k), not unequal size.

### Measured 2026-09-20 — the metric change works

Re-ran all three on GPU and scored the **validation** split (102,501 hands).
Script and log: scratchpad `sets_balacc.py` / `run.log`. `sets.ipynb` was not modified.

| Model | Plain acc | **Balanced acc** | Classes never predicted |
|---|---|---|---|
| Flattened FFN | 99.40% | **72.32%** | 3 |
| Deep Sets | 84.39% | **35.80%** | 3 |
| Set Transformer | 99.82% | **77.78%** | 3 |
| Majority ("Nothing") | 50.17% | **11.11%** | 9 |

Per-class recall (val support in brackets):

| class | support | FFN | Deep Sets | Set Transformer |
|---|---|---|---|---|
| Nothing | 51,427 | 99.98% | 87.67% | 100.00% |
| One pair | 43,432 | 99.77% | 87.37% | 100.00% |
| Two pairs | 4,764 | 96.94% | 51.53% | 100.00% |
| Three of a kind | 2,133 | 93.58% | 39.94% | 100.00% |
| Straight | 394 | 93.91% | 36.04% | 100.00% |
| Flush | 180 | **0.00%** | **0.00%** | **0.00%** |
| Full house | 150 | 86.67% | 14.67% | 100.00% |
| Four of a kind | 20 | 80.00% | 5.00% | 100.00% |
| Straight flush | 1 | **0.00%** | **0.00%** | **0.00%** |
| Royal flush | **0** | n/a | n/a | n/a |

**What this buys the notebook.** On plain accuracy the Set Transformer beats the FFN by
0.4 points — noise. On balanced accuracy it beats it by **5.5 points**, and the per-class
column shows why: the Set Transformer holds 100% recall on seven of the eight testable
classes while the FFN degrades as classes get rarer (93.6% → 86.7% → 80%). That is the
inductive-bias lesson, and it needs no retraining to tell.

**Two things to write into the notebook:**
- **Every model scores 0% on flush** (180 examples) and on the lone straight flush. The
  Set Transformer's entire 0.18% error budget is essentially flushes misread as something
  else. Flush is the one class defined purely by a property of the *suit* set, with no rank
  structure to fall back on — a genuinely interesting observation, not a footnote.
- **Royal flush has zero validation examples**, so balanced accuracy is a mean over nine
  classes. That is also why the majority baseline scores 11.11% and not 10%. State it.

**Caveat — Deep Sets did not reproduce.** 84.39% val here against 92.1% committed, same
seed path and same 200k steps; final val loss 0.30 vs 0.142. The committed run was CPU and
this one GPU, and Deep Sets sits on a long plateau where float non-determinism shifts the
escape point. Its 35.80% therefore comes from an unluckier run than the committed one.
**Do not publish 35.80% as the Deep Sets number without a rerun** — though the qualitative
story is unaffected, since the rare-class gap is 80–100 points against an 8-point gap in
plain accuracy.

- [x] **2.10 closed 2026-09-20: the UCI URL still resolves.** The pre-restructure path
      `archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-{training-true,testing}.data`
      returns HTTP 200 with no redirect, and row counts (25,010 + 1,000,000) match the
      notebook exactly. Cell 4 needs no change. If it ever breaks, the modern path
      `archive.ics.uci.edu/static/public/158/poker+hand.zip` also works
- [ ] **State the step-budget asymmetry explicitly** rather than equalising it — the review
      allows either, and no-retraining means we take this branch. Deep Sets had 200k steps
      against 50k, so cell 35's "DeepSets takes much longer to converge" is a claim the
      design cannot support; say plainly what each model got (2.1)
- [ ] **The central change (2.2): per-class recall plus the macro average**, alongside a
      majority-class baseline row. Note the terminology — *micro*-averaged accuracy is
      identical to plain accuracy for single-label classification and would still read
      99.5%; the **macro** average (unweighted mean of per-class recall = balanced accuracy)
      is what punishes a model that never predicts the rare hands. Royal flush is ~0.0015%
      of hands, and nothing + pair + two pair + three-of-a-kind is 98.9%, so 99.5% accuracy
      is consistent with failing every rare class. Requires no retraining — the checkpoints
      already exist
- [ ] Per-hand permutations, not one shared permutation for all 1000 hands — **and state
      that 0% is a tautology for Deep Sets and the Set Transformer, so the test is a unit
      test on the implementation. That is the lesson** (2.3)
- [ ] ~~Train on 5 cards, evaluate on 7 (2.4)~~ — **dropped** by Yoav 2026-09-20: the
      dataset is five-card hands, so a 7-card evaluation has no data behind it. The
      "variable-size input ✓" claim in cell 35 must therefore be **softened or removed**
      rather than demonstrated — do not leave an asserted capability the notebook never
      shows
- [ ] Hedge the Zaheer universality statement (2.5)
- [ ] Fix cell 25's display math (`\\` line breaks) (2.6)
- [ ] Consistent train/val/test accuracy subsets (2.7); fix reversed `split` operands (2.8)
- [ ] Add exercises (2.9)

---

### WP5 — `bpe-tokenizer.ipynb` and `bpe.py`  ✅ done 2026-09-24 (branch `wp5-bpe-tokenizer`, `90dc41f` → `ba0d369`)
Does A3 (data scale), 7.1–7.7, 7.10 — **minus the rows already closed in WP0** (7.3, 7.9).
**WP6 depends on this package**: under the code reuse contract `nanochat.ipynb` imports the
`bpe.py` that this notebook generates. **E5: no biology content this pass** —
Exercise 4 stays on Spanish/French but must become answerable (see 7.10).

- [x] Default corpus is `TinyStoriesV2-GPT4-valid.txt` (22.5 MB); the train split is an
      explicitly flagged option. Do not hold two copies in memory (7.1, A3)
- [x] Vocab sweep on a fixed subsample; rewrite or drop the duplicate Exercise 1 (7.2)
- [x] Save the tokenizer to `checkpoints/` (7.3, done under WP0 — verify here)
- [x] Put the ~30-line merge loop **inline**, built cell by cell with prose; `%pycat` opens
      a pager and shows the student nothing (7.4). Per the code reuse contract those cells
      also **generate `bpe.py`**, which every downstream notebook and `nanochat_chat.py`
      imports; merges+vocab go to `checkpoints/bpe_tokenizer.pkl`
- [x] Add the tokenizer-invariance section: cross-entropy per token is not comparable
      across tokenizers, which is why bpc is the reporting unit (7.5, A4)
- [x] **7.10 (S1):** `bpe_encode` silently maps unknown characters to token id 0. Either raise
      or add an explicit `<unk>` with a reported rate; then fix cell 2's OOV table (7.6) and
      Exercise 4, which currently asks students to count a fallback that does not exist.
      Name byte-level BPE (Exercise 2) as the real answer to OOV
- [x] Compression ratio on held-out stories, not `stories[:200]` from training (7.7)
- [x] Exercises must not duplicate `nanochat.ipynb`'s (7.8 deferred, 8.17)

---

### WP5 outcome, and what it settled for later packages

**The module emission mechanism is decided and tested** (the plan listed it as
undecided and blocking). The notebook defines each function normally — so the kernel
has it and the reader sees it — and one cell writes `bpe.py` from
`inspect.getsource` over an ordered tuple of the *live* definitions, then
`importlib.reload`s it and round-trips a string as a check. Reading current bindings
rather than appending text is what makes it idempotent and order-independent: the cell
rewrites the whole file every time, so re-running it, or running the cells above out of
order and regenerating, yields the same module. `%%writefile -a` was rejected for
exactly the duplication/scrambling trap the plan named. **Verified end to end under
`nbconvert --execute`, not just in an IPython shell.** Use the same shape for
`nanochat_model.py` in WP6.

**Decisions taken by Yoav 2026-09-24:**

| Question | Decision |
|---|---|
| 7.10 OOV policy | **Raise on unknown characters.** No `<unk>`, no vocab-size change, no forced retokenise from this package. Byte-level BPE is named as the real fix and is Exercise 2. |
| Generated modules in git | **Keep `bpe.py` tracked and commit it.** A student run dirties the tree; the generated header explains why. Same rule for `nanochat_model.py`. |
| Commit attribution | **`CLAUDE.md` wins — no `Co-Authored-By`/`Claude-Session` trailers.** The contradictory session config is overridden. WP4's commits need no amending. |
| `nanochat-review.md` | **Stays untracked**, by Yoav's call. |

**Two things WP5 measured that later packages depend on** (full numbers in `runs.md` T1):

- **The shipped `checkpoints/bpe_tokenizer.pkl` was trained on the *train* split** — 227
  single-character tokens against the valid split's 88, and **0 of 1024 ids in common**.
  This answers WP-T's verification row. Consequence: moving the default corpus to the
  valid split, which WP5 has now done, **invalidates every existing nanochat, SFT and
  GRPO checkpoint**. They are not merely stale, they are keyed to a different id→string
  map, and decoding with the new tokenizer produces fluent nonsense with no error. The
  old tokenizer is preserved as `checkpoints/bpe_tokenizer_trainsplit.pkl` (gitignored).
  **`nanochat-chat.ipynb` cannot be re-run meaningfully until WP6–WP8 retrain.** That is
  the price of the corpus change and it was already implied by WP6's planned retrain,
  but it is now real rather than prospective.
- **A closed character vocabulary's coverage is an accident of the corpus.** The 88
  characters include `é` and `ñ` but not `ß`/`ä`/`ü`, so Spanish encodes and German
  refuses. A draft of the notebook prose asserted the reverse and the first execution
  falsified it — the fifth instance in this project of prose written ahead of the
  measurement. The generalisable form of "ceilings before blame": **anything about which
  inputs a tokenizer covers is cheap to compute and must not be reasoned about.**

**Not done here, deliberately:** `<|endoftext|>` (8.11) is untouched, and `uint16` token
storage is untouched. Both remain WP-T's, and WP-T's corpus question (which corpus
*pretrains* nanochat, as distinct from which trains the tokenizer) is still open.

---

### WP-T — Token pipeline  ⬜ not started
**New package, inserted between WP5 and WP6.** Not in the review; added because four
scattered changes all invalidate the same artifacts and must be decided and executed once.

The problem: `checkpoints/bpe_tokenizer.pkl`, `train_tokens.npy` (1.06 B ids) and
`nanochat_checkpoint.pkl` (embedding rows indexed by those ids) form one chain. Any of the
following breaks all three — **even if 8.1 is not confirmed**:

- 7.1, moving the tokenizer's default corpus to the valid split
- 7.10, adding `<unk>`
- 8.11, adding `<|endoftext|>`
- 8.1, the QK-norm change (params only, but it forces the retrain anyway)

Tasks:

- [x] **Decided by Yoav 2026-09-24: BPE trains on ~10% of the train split; nanochat
      pretrains on the *full* train split.** They differ, which is the normal arrangement —
      merge frequencies converge on a sample long before the model has seen enough data.
      The valid split is ruled out for pretraining (~10.9 M tokens = ~17 epochs at the
      current budget, i.e. memorisation). `data/TinyStoriesV2-GPT4-train.txt` is now on
      disk (2,226,845,268 characters; disk at 97%, 34 GB free).
      **This requires the `chars=` fix below — without it the plan does not work.**
- [x] **The one-line fix that makes the 10% sample safe — shipped in WP5.** `bpe_train`
      now takes `chars=`, overriding the character inventory:

      ```python
      chars = sorted(set(full_corpus))            # one cheap pass, no merging
      vocab, merges = bpe_train(sample, vocab_size=1024,
                                special_tokens=(END_OF_TEXT,), chars=chars)
      ```

      **Why it is needed, measured on the real corpus:** the full train split has **228
      distinct characters**; a 10% sample sees **158** and misses **70**. Those 70 cover
      **296 occurrences out of 2.23 billion** (1.3e-7) — negligible by frequency and fatal
      under raise-on-unknown, since one stray character aborts the encode of the whole
      corpus. Merge statistics converge on a sample; character inventories cannot, because a
      character occurring once is either in the sample or it is not.
- [x] **Corpus cleaning — done 2026-09-24, and it lives in the notebook.** Yoav's steer:
      this is teaching material, not a pipeline step, so `clean_corpus` is built and
      explained in `bpe-tokenizer.ipynb` (Step 4) and emitted into `bpe.py`. **WP6 must call
      the same function on the pretraining corpus** — clean differently and the corpus holds
      characters the vocabulary has no id for.
      Policy: keep a character if it occurs ≥ `min_count` (100) **or** is printable ASCII;
      drop any document using anything else, **never edit text**. On the train split that is
      104 characters kept and **+124 merges for 389 documents of 2,717,495 (0.014%)**; on the
      valid split the notebook actually runs, 79 kept and +9 merges for 109 of 27,630
      (0.39%). Full numbers in `runs.md` T3/T4.

- [x] ~~Verify whether the shipped `checkpoints/bpe_tokenizer.pkl` was trained on the train
      or the valid split.~~ **Answered in WP5: the train split** (227 single-character
      tokens vs the valid split's 88, 0/1024 ids shared). WP5 has since overwritten it
      with a valid-split tokenizer, so **the retokenise is forced** — see WP5 outcome above
- [x] ~~New special tokens go at the **end** of the vocab, never inserted at id 0~~
      **Done in WP5**: `bpe_train(special_tokens=...)` appends after the merge loop, so every
      ordinary token keeps the id it would otherwise have had
- [x] **8.11 done in WP5** (`537c83b`), and it was smaller than budgeted here because two
      of this row's premises were wrong. **The vocabulary does not grow to 1025**: the token
      is reserved *out of* the 1024 budget (88 characters + 935 merges + 1 special,
      `<|endoftext|>` at id 1023), so embedding and head shapes are unchanged and no new
      shapes or retrain follow from 8.11 itself. **And the `\S+|\s+` segmenter does not
      shred the literal** — `<|endoftext|>` contains no whitespace, so `\S+` matches all of
      it; the demo cell prints this. The real argument for special handling is that a segment
      is not a token: without a reserved id the separator would be encoded character by
      character and merged like an ordinary word, and here it would raise outright, since
      `<`, `|` and `>` never occur in TinyStories. Cost measured: one merge, held-out
      compression 2.1065× → 2.1061×. **The retokenise is still required**, but by the corpus
      change, not by this row
- [ ] **Store tokens as `uint16`, not `int64`.** Today cell 40 does `jnp.array(train_data)`,
      putting 7.65 GB of train tokens plus 0.85 GB of val on a 16 GB card — around 12–13 GB
      total with params, Adam state and materialised attention weights, against XLA's
      default 75% preallocation of 12.3 GB. A retrain is one batch-size bump from OOM.
      Two-line change, 8.50 GB → 2.1 GB
- [ ] Retokenise **once**, producing exactly one new `train_tokens.npy` under `checkpoints/`.
      Budget 2.23 GB download + 1–3 h of pure-Python `bpe_encode` if the train split is used
- [x] **Disk handled 2026-09-24.** `checkpoints/train_tokens.npy` (8.5 GB, `int64`,
      `(4132598, 257)`) **deleted** — it was tokenised against the 227-character train-split
      tokenizer WP5 replaced, so its ids no longer mean what the current tokenizer says. It is
      gitignored and regenerated by `nanochat.ipynb`; as `uint16` it will come back at 2.1 GB.
      Free space **34 GB → 41 GB**, repo 17 GB → 8.0 GB, `checkpoints/` 8.7 GB → 728 MB. The
      repo-root duplicates this plan listed were already gone (WP0). Nothing else in the repo
      is stray: `.pixi` 5.6 GB, `data/` 2.2 GB (the train split), `.git` 250 MB.
      **Not deleted, needs a decision:** the six nanochat/SFT/GRPO checkpoints (606 MB) are now
      tokenizer-mismatched and WP6–WP8 will retrain them, but they are the only record of those
      runs and cost hours of GPU to reproduce.

---

### WP6 — `nanochat.ipynb`  ⬜ not started
**Depends on WP5** (it imports the `bpe.py` that `bpe-tokenizer.ipynb` generates).

- [ ] **Clean the pretraining corpus with `clean_corpus`, at the same `min_count` the
      tokenizer was built with, and assert it.** This is a hard coupling, not a convention:
      clean with a different threshold and the corpus contains characters the vocabulary has
      no id for, and `bpe_encode` raises **partway through encoding 2.23 GB** — an hour in.
      Cheap to hit, tedious to debug. Check `set(corpus) <= set(vocab)` before starting the
      encode, not during it. The tokenizer pickle does not record `min_count`, so either
      re-derive the inventory from the saved vocabulary (`[t for t in vocab if len(t)==1]`)
      or pass the threshold explicitly.
**Step 1 happens before any edit.**

- [ ] **8.1 is already confirmed by arithmetic, so the entropy measurement is not a gate.**
      Verified in the code: RoPE, then `Q /= ‖Q‖₂`, `K` likewise, then `/sqrt(64)`. Logits
      land in [-0.125, 0.125]; spread 0.25; max ratio between two softmax weights
      e^0.25 = **1.284**. With `seq_len=256` every weight sits within ±13% of 1/256.
      Attention is a near-uniform mean of V **by construction**. Do not spend an hour
      measuring what the arithmetic already proves.
- [ ] **The decision-relevant experiment is a short A/B**: ~2,000 steps of each variant,
      ~15 min apiece, compared on val loss. Run that before committing to a 4–7 h retrain.
- [ ] Temper the expected result in the prose: the current setup is *degenerate*, not inert.
      The ±13% deviations still carry gradient and the residual stream still carries the
      current token, so the model is "current token + bag of context" — which is why it
      still reached 1.069 nats/token. Expect a real but not miraculous improvement.
- [ ] **Decide: does the RMS QK-norm get a learnable per-head gain?** It changes
      `init_params`, the parameter count and the checkpoint schema. Under-specified today.
- [ ] Then replace the L2 normalisation with RMS-style per-head norm, keep the
      `1/sqrt(head_dim)` scale, retrain, and regenerate the SFT and GRPO checkpoints.
      This re-opens WP7 and WP8.
- [ ] Keep the entropy / attention-map figure as **teaching material** (8.10), decoupled
      from the decision.
- [ ] Make the RoPE prose match the RoPE code — prose is interleaved, code is split-half (8.2)
- [ ] Generate the scaling table from `param_formula` / `init_params`; cell 26 contradicts
      cell 27's own output and contradicts itself (8.3)
- [ ] Read vocab size and chars-per-token from the loaded tokenizer: 1024 and ~2.2, not
      512 and "3–4" (8.4)
- [ ] Fix cell 50's table: vocab 1024, 26.2M params, `optax.adamw` (8.5)
- [ ] "nats per token", not "bits-per-token"; fix the `\text` typo (8.6)
- [ ] Re-run end to end or strip: committed outputs are a composite of several runs (8.7)
- [ ] **Explicit `RESUME` flag, default off (8.8) — do this FIRST, before the A/B.**
      `resume_from = checkpoint_path` will otherwise silently load the old L2-normed
      checkpoint into the new architecture and, because the pytree shapes are identical,
      will not even error. WP6 owns this item, not WP0
- [ ] Coordinate with **WP-T**: the tokenizer/separator/`uint16`/retokenise decisions live
      there, and 8.11 is executed there, not here
- [ ] **A3 for this notebook:** cell 32 repeats the 2.23 GB train-split download and holds
      the corpus twice. Default to `TinyStoriesV2-GPT4-valid.txt` (22.5 MB) like WP5, with
      the train split an explicitly flagged option. Without this, WP0's acceptance criterion
      that everything agrees on which split is used cannot be met
- [ ] **B5 for this notebook:** fixed step budget, no early stopping; keep best-checkpoint
      *saving* (that is how class-time checkpoints are produced). `nanochat.ipynb` is one of
      the six training notebooks B5 covers and has no Part C ID of its own
- [ ] Held-out bpc + WP1 baselines (8.9, A4)
- [ ] Attention-map figure across layers for a TinyStories sentence (8.10)
- [ ] `generate`: hoist the RoPE tables and mask out of the per-token loop, **and jit it**
      (both halves of 8.12; KV caching / 6.5 is deferred, so nothing else covers this)
- [ ] Duplicate "## Data Preparation" headings (8.13); bare `except:` (8.14); perplexity
      example that matches the real val loss (8.15); FLOPs clause (8.16)
- [ ] Exercise 4 duplicates bpe Exercise 1 — assign to one notebook (8.17);
      re-scope Exercise 7 if WP2 already introduced `scan` (8.18)
- [ ] Per the code reuse contract, the model cells **generate `nanochat_model.py`**; sft,
      grpo **and `nanochat-chat.ipynb`** import it and stop re-pasting the model.
      `load_checkpoint` returns a **dict** everywhere (B2, 8.x, 9.3, 10.9).
      **Four pasted copies now, not three** — see WP-C
- [ ] **From WP-C, inherited from the deleted `nanochat_chat.py`:** reconcile `generate`
      with the `top_k`/`top_p` sampling that now lives in `nanochat-chat.ipynb`, so
      `nanochat_model.generate` is the one implementation. WP-C already did the
      `generate` half of **8.12** (fixed-shape buffer, hoisted RoPE/mask, jitted) —
      port that implementation rather than re-deriving it

---

### WP7 — `nanochat-sft.ipynb`  ⬜ not started
Depends on WP6 (and on its retrain, if 8.1 is confirmed). Task decided: **TinyStories-Instruct**.

- [ ] Replace story-continuation with `roneneldan/TinyStories-Instruct` — the pretrained model
      demonstrably fails it, so the before/after in §5 shows a real difference (9.1)
- [ ] Replace the pasted model cell with `from nanochat_model import ...` (B2, 9.3)
- [ ] Honest fixed budget with a **stated target** — the committed run is ~80 steps over 640
      of 24,124 examples, and its val improvement (0.985 → 0.870) was measured on a **single
      random batch of 8**, i.e. inside the noise. Fix the val set and do a full pass; say how
      many steps and why (9.4, and the same disease as 10.13 one notebook earlier)
- [x] **Verified 2026-09-20.** The HF repo is `roneneldan/TinyStoriesInstruct` — **no
      hyphen in the repo name**, unlike the review's `TinyStories-Instruct`, which 404s.
      Files inside it *are* hyphenated:
      `TinyStories-Instruct-train.txt` (2.66 GB) and `TinyStories-Instruct-valid.txt`
      (26.9 MB). Direct URL pattern:
      `https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct/resolve/main/<file>`
      Use the **valid** file (26.9 MB) as the default, consistent with the corpus rule
      elsewhere; no `datasets` dependency needed since these are plain text
- [ ] Add the valid file to `download_data.py`
- [ ] Format, for the prompt template: each record has `Features:` (optional — tags like
      Dialogue, BadEnding, MoralValue, Foreshadowing, Conflict), `Words:` (3 vocabulary
      items the story must contain), `Summary:`, `Story:`, and sometimes a random sentence
      that must appear verbatim. The pretrained model demonstrably cannot satisfy these
      constraints, which is the whole point of the swap (9.1)
- [ ] **Decide the chat format here, because WP8 depends on it.** GRPO hard-codes
      `[INST] … [/INST]`; TinyStories-Instruct ships `Features:/Words:/Summary:/Story:`.
      If SFT trains on one and GRPO prompts with the other, GRPO starts from a policy that
      has never seen its own template
- [ ] The committed outputs are from **two machines** — cell 2 shows `JAX 0.9.2 | cpu` with
      a macOS path while cells 13/16 are CUDA. 8.7 catches this for nanochat; nothing
      catches it here
- [ ] Re-run so the key comparison cell actually has output (9.5)
- [ ] State explicitly that SFT uses the valid split because pretraining used train (9.6)
- [ ] Two sentences on per-example vs per-token loss normalisation (9.7)
- [ ] Keep Exercise 3; consider promoting it to a two-cell ablation (9.8)
- [ ] **B6 consequence:** state at the top that this notebook requires
      `checkpoints/nanochat_checkpoint.pkl`, which is not distributed — the student must
      produce it by running `nanochat.ipynb`, and roughly how long that takes
- [ ] 9.9 (the `apply_rope(Q,c,s)/norm(Q)` reader trap) needs no work here — the line lives
      in the pasted model cell, which the import above deletes. Confirm it is gone rather
      than editing it

---

### WP8 — `nanochat-grpo.ipynb`  ⬜ not started
Depends on WP7. Decided: **inner epochs K>1** so clipping engages.

**Cost reality, from the measured 1 h 42 m run.** 98% of wall-clock is `sample_group`:
un-jitted, no KV cache, one batch-1 forward per sequence per token (G=4 separate calls),
`.item()` syncing every token, and a fresh XLA compile at every new sequence length (1072
compiler log lines sit in cell 11's committed stderr). That is the 55 ms/forward. Evaluation
is already 44% of the training cell. The plan's additions (two more baselines, an N=4 axis,
a reward distribution, a clipped-fraction plot, a larger `n_eval`) push this to **8–12 h**
unless sampling is fixed first.

- [ ] **Highest-leverage change in the entire plan, and it was not in the review: batch
      `sample_group`.** Put the G sequences into one `(G, T)` call, pad to fixed `T`, jit it,
      and drop the per-token host sync — worth roughly **20×**. A KV cache is another ~5×.
      Note that 10.15 ("pad and jit `grpo_loss`") optimises the ~600 batched loss/grad
      forwards, i.e. about **2%** of runtime. Do this one first.
- [ ] Reconsider the WP2 deferral of KV caching (6.5): WP8 inherits its absence as cost.

- [ ] Add K>1 gradient epochs per rollout; plot the clipped-token fraction (10.3).
      **Verify the fraction is actually > 0.** At `lr=1e-5` the clip needs a per-token
      log-prob shift of 0.18, which a handful of AdamW steps on a 26 M model will not
      produce — the plot will read 0.000 and the notebook will have added machinery that
      still does nothing. Be prepared to raise `lr` or K, or shrink `clip_eps`, to make the
      demonstration real; otherwise fall back to the review's honest option (b) and state
      that the objective is vanilla policy gradient with a group baseline. Note too that
      K>1 with G=4 and one prompt per step means K passes over 4 stale samples: high
      variance, with only the KL term against `ref_params` holding it
- [ ] Genuinely held out: train on N ∈ {2,3}, evaluate also on N=4 — or delete the
      "held-out" claim, which is currently false (10.1). **Do 10.4 first:** at `max_new=80`
      tokens, four TinyStories sentences is ~50–65 tokens, so an N=4 evaluation would
      largely measure truncation rather than instruction-following
- [ ] Baselines: the SFT model's reward and the reward of an instruction-ignoring sample.
      0.438 means nothing without them (10.2)
- [ ] Fix the reward's truncation artefact — generate to a stop condition, or count only
      complete sentences and say so (10.4)
- [ ] Plot the zero-variance-group skip fraction (10.5)
- [ ] One sentence + reference on group-std normalisation bias (Dr. GRPO, Liu et al. 2025) (10.6)
- [ ] De-duplicate `eval_reward` / `evaluate` (10.8)
- [ ] Replace the pasted model cell with `from nanochat_model import ...`; drop the
      `make_rotary` alias and the inline `jnp.triu` mask (B2, 10.9)
- [ ] State wall-clock and the GPU requirement at the top (10.10) — measured 1 h 42 m today,
      and re-measure after the sampling fix
- [ ] Replace the cherry-pickable SFT-vs-GRPO pair with a reward distribution (10.11)
- [ ] **Fix the evaluation properly (10.13).** The review's SE of 0.055 is arithmetically
      right but understated: the 80 samples are not independent (4 completions share each
      prompt), so with intra-prompt correlation ~0.3 the true SE is nearer **0.075**. And
      the task space is *exactly 20 prompts* (10 animals × `TARGET_SENTENCES=[2,3]`) sampled
      with replacement, so the prompt set barely changes — the dominant noise is the
      **sampling keys**, not the prompt set. The clean fix is to **enumerate all 20 prompts
      deterministically** and raise `G_eval`. Fixing the seed is still worth it, but for
      common random numbers across checkpoints, which is the real argument
- [ ] Eval runs at `temperature=0.7` while rollouts run at 0.9, unremarked. Reconcile or say why
- [ ] With SE ≈ 0.075 against an observed range of 0.350–0.438, **best-checkpoint selection
      is selecting noise.** Either report the final checkpoint, or state plainly that "best"
      means best under a 0.075-SE estimate
- [ ] Remove early stopping — it runs on a noisy metric over a shifting sample with a
      threshold two orders of magnitude below its SE (10.14, B5)
- [ ] Pad to a fixed length, then jit `grpo_loss`; a sentence on shape stability in JAX (10.15)
- [ ] Keep the "Sketch: Reasoning Models" section (10.12)
- [ ] **B6 consequence:** same required-checkpoint statement at the top, naming
      `checkpoints/nanochat_sft_checkpoint.pkl` and the notebook that produces it

---

### WP-C — `nanochat-chat.ipynb`  ✅ done 2026-09-20 (branch `wp-c-chat-notebook`)
**New package, not in the review.** Yoav 2026-09-20: convert `nanochat_chat.py` into a
notebook, placed after GRPO and before the agent notebook. The script is deleted.

Why it earns a slot in the arc: it is the only place the three checkpoints are compared
against each other, and inference has two teachable ideas that appear nowhere else in the
course — sampling (temperature / top-k / top-p) and JAX shape stability.

- [x] `nanochat-chat.ipynb` created, 26 cells, executed end to end on GPU with real outputs
      committed. Sections: load checkpoints → model (pasted) → sampling → the real
      next-token distribution → generation → three-way comparison → sampling knobs →
      interactive `chat()` → what is missing → 5 exercises
- [x] `nanochat_chat.py` deleted; `index.ipynb` gains an **Inference** section between GRPO
      and SWE Agent; `CLAUDE.md` notes the notebook is inference-only
- [x] **Fixed a real performance bug inherited from the script.** The script's `generate`
      grows the context by one token per step, so XLA recompiles on *every token*: measured
      **429 s** for 60 tokens on the first call, ~3.4 s (≈57 ms/token) once those lengths
      were cached. Rewritten to a fixed-size `seq_len` buffer with the RoPE tables and mask
      hoisted out of the loop and `next_token_logits` jitted — **4.6 s first call, 0.2 s
      after, ≈3 ms/token**. Verified byte-identical output against the naive version.
      This closes the notebook half of **8.12** ahead of WP6
- [x] Measured three-way result, committed in the notebook: fraction of 20 samples with
      exactly 3 sentences — pretrained **4/20**, SFT **9/20**, GRPO **12/20**. Monotone,
      but SE ≈ 0.11, so **SFT vs GRPO is inside the noise**; the notebook says so and
      Exercise 3 makes the student check it
- [ ] **WP6 consequence: there are now FOUR pasted copies of the model**, not three
      (`nanochat-sft`, `nanochat-grpo`, `nanochat-chat`, plus the original). WP6's
      "replace the pasted model cell with `from nanochat_model import ...`" must cover
      `nanochat-chat.ipynb` too. The notebook's own prose promises this fix, so leaving it
      out would make the notebook wrong
- [ ] **WP7 consequence:** this notebook hard-codes `[INST] … [/INST]` in `chat()`, in both
      comparison prompts and in the sentence-count measurement. If WP7 moves SFT to the
      TinyStoriesInstruct `Features:/Words:/Summary:/Story:` format, **all of them must
      change and the notebook must be re-run**
- [ ] **WP6/WP-T consequence:** section 8 states there is no `<|endoftext|>` token and that
      this is why samples end mid-word. If 8.11 lands, that paragraph and the truncation
      caveat in section 5 both need rewriting
- [ ] Exercise 4 asks the student to implement a KV cache. That is the deferred 6.5 / 8.12
      remainder — if a later package implements one, reconcile so the exercise is not
      solved in the repository

---

### WP9 — `minisweagent.ipynb`  ⬜ not started
Unblocked. Least work needed; the risks are operational.

- [x] **`qwen3.5:9b` confirmed published** (2026-09-20). Add **`qwen3.5:4b`** (3.4 GB) as
      the low-VRAM fallback; `2b` and `0.8b` also exist if a room is really constrained
      (11.1, 1.6)
- [ ] `AUTO_APPROVE` flag so the notebook survives "Run All"; document the interactive
      default as a deliberate choice (11.2)
- [ ] Move the least-privilege warning **before** Task 1 (11.3)
- [ ] Working prompt-injection demonstration, built on the Task 3 pipeline (11.4 + 11.10)
- [ ] Replace `messages[-2]` with an explicit search for the last tool output, and delimit
      the interpolated content (11.10)
- [ ] Date and attribute the "~74% on SWE-bench Verified" claim; note the local 9B model
      will do far worse (11.5)
- [ ] Count tokens with the workshop's own tokenizer instead of `chars // 4` (11.6)
- [ ] Rewrite Exercise 1 Q5 around the `ValueError` path — `parse_action` never returns
      `None` (11.9)
- [ ] Update the model names in cell 27 (11.8). "Claude Sonnet 4.6" is out of date — the
      current Claude line is the **Claude 5 family** (Opus 5, Sonnet 5, Fable 5.1) plus
      Haiku 4.5. Re-check the other vendors' names at delivery
- [ ] Note the `export.arxiv.org` network requirement; cached fallback for offline rooms (11.11)

---

### WP10 — `index.ipynb` rewrite  ⬜ not started
Last: it describes the result of everything above. Does 1.7 (1.1 is closed in WP3 by
deleting the table). `index.ipynb` becomes navigation and orientation only — no results.

- [ ] Learning outcomes
- [ ] Session arc: sets → recurrence → attention → tokenisation → pretraining → alignment → agents,
      saying what each step adds
- [ ] Time estimates, prerequisites, CPU/GPU annotation per notebook

---

## Exercise solutions

`solutions/` does not exist yet, and with the LR-sweep session retired (2026-09-21)
nothing is scheduled to create it. Suggested convention,
to be fixed once the first one lands: `solutions/exercise-<n>-<slug>.py` for the script,
`.md` for the written answer, `.json` for raw measured numbers so the prose can be
checked. Nothing in `.gitignore` touches `solutions/`, so all three are tracked.

**Solutions must never write to `checkpoints/charlm_result_*.json`.** Those are the
committed rows of the comparison table; a sweep that runs through the notebooks
overwrites them and the diff looks like a legitimate result update.

---

## Branch state (2026-09-24)

Neither revision branch has been merged to `main`, and they are **stacked**:

```
main
 └── sets-revision      WP4, 6 commits (c827495 → 6af3a06)
      └── wp5-bpe-tokenizer   WP5 + WP-T, 7 commits (90dc41f → ba0d369)
```

So `wp5-bpe-tokenizer` carries WP4's commits too, and pushing or merging it brings both.
The ground rules call for one PR per work package; that would mean merging `sets-revision`
first, then `wp5-bpe-tokenizer` on top. **Yoav's call whether to keep them separate or land
them together.**

---

## Open items needing Yoav

**Raised 2026-09-24, both trivial but both unresolved:**

- [x] **Resolved 2026-09-24: `CLAUDE.md` wins — no trailers.** WP4's four commits stand as
      made; WP5's follow the same rule.
- [x] **Resolved 2026-09-24: stays untracked**, by Yoav's call. Original note: **`nanochat-review.md` is still untracked** — along with `slurm-download-data.sh` and an
      empty `.codex`. The review is the document every finding ID in this file refers to, and
      an untracked file is exactly the blind spot that hid `plan.md`'s eleven-commit drift.
      Recommend committing it.
- [x] **Delivered 2026-09-24:** the WP4 class-imbalance handoff is filed as
      `yoavram/DataSciPy` **issue #15** (FFN-only session, all numbers, reusable ceiling code,
      caveats). Nothing further needed here unless that agent comes back with questions.


- [x] **Resolved 2026-09-24: keep them tracked and commit them.** A student run dirties
      the tree; the generated header explains why. Applied to `bpe.py` in WP5, and the same
      rule governs `nanochat_model.py` in WP6.
- [x] **Resolved 2026-09-24: BPE on ~10% of the train split, nanochat pretrains on the
      full train split.** See WP-T; needs the `chars=` fix, which WP5 shipped.
- [x] **Resolved 2026-09-24: add it.** Done in WP5 (`537c83b`) and it did not force
      anything on its own — the token is reserved out of the 1024 budget, so shapes are
      unchanged. The retokenise it shares is the one the corpus change already forced.
- [ ] **RMS QK-norm:** learnable per-head gain, or not? Changes the checkpoint schema.

- [ ] **Checkpoint distribution** (B6) — deferred by Yoav. Revisit before delivery: without
      it, `nanochat-sft` and `nanochat-grpo` cannot be run by a student.
- [ ] Verify the non-Claude model names in `minisweagent.ipynb` cell 27 at delivery (11.8)
- [ ] **GRU layer-norm ablation (5.2)** — keep the broken post-gate LN as a two-cell
      ablation alongside the fix? The review rates it a good lesson. Yoav's call.
      **Still un-ablated as of 2026-09-21**, and the plan's own rule was to run the ablation
      before writing the prose.
- **C6/C7 notebook prose — delegated 2026-09-21, tracked as its own issue.** Not this
  file's business any more. For context: `4abebda` harvested both sweeps but left the prose
  alone, so `text-transformer.ipynb`'s conclusion still claims the depth curve is "still
  descending" at 6 where C7 shows it turns over. It is the one factually wrong claim
  currently committed, and it is gated on WP-N.
- **LR sweep (exercise 2) — retired 2026-09-21.** Handed off after `57b55df` and never run;
  the sessions since went to depth and context sweeps, which answered the scaling question
  it was meant to probe. `handoff-lr-sweep.md` **deleted** in this pass — it had also gone
  stale on branch state, still describing `wp1-shared-experiment` as unmerged. If the sweep
  is ever wanted, it starts from `runs.md`, not from that file.
- [ ] **`solutions/` still does not exist.** It was to be created by the LR-sweep session,
      which is now retired, so nothing is scheduled to create it. Decide who does when the
      first exercise solution is written.

---

## Corrections to the review (from the feasibility audit)

**From WP4, 2026-09-24 — three review items were wrong or misleading as written:**

- **2.6 is described backwards.** The review says cell 25 "has `\\` line breaks that render
  wrong". The actual defect was three equations sharing one `$$…$$` with **no** `\\` at all,
  rendering on one line. Fixed with an `aligned` environment.
- **2.3's premise did not survive measurement.** The review expects per-hand permutations to
  expose more FFN order-dependence than one shared permutation. Measured: **1.00% per-hand vs
  1.10% shared** — the stronger test *agrees*. The fix is still right (the old test sampled 1
  of 120 orderings) but must be reported as agreement, not discovery. An exercise resting on
  the opposite expectation was replaced.
- **2.8 is not cosmetic.** The reversed `split` operands swapped which key continued the chain,
  so fixing it shifted the key stream for two of the three models and moved every number in the
  notebook. Any package fixing a `split` should expect to re-execute, and should take prose
  numbers from the notebook's own run rather than from a side script.


Record these so a fresh session does not re-derive them or write prose against a claim that
does not hold.

1. **8.1 QK-norm is confirmed** — arithmetic verified against the code and the checkpoint
   config. But the prescribed entropy *measurement* is not a decision gate; it is provable
   on paper. Use a 2,000-step A/B instead, and keep the entropy figure as teaching material.
2. **5.2 GRU layer-norm is overstated.** LN is idempotent and the incoming `h` is already
   normalised, so the near-identity carry path survives at `z → 0`. Downgrade **S1/S2 → S2**.
   Run the ablation before writing the prose.
3. **10.3 clipping is inert — correct.** But K>1 will probably not visibly fix it at
   `lr=1e-5`; the clipped fraction will read 0.000 without further changes.
4. **10.13's SE of 0.055 is understated** (~0.075 once intra-prompt correlation is
   accounted for), and the diagnosis is off: the prompt space is only 20 items, so the
   dominant noise is the sampling keys, not a shifting prompt set.
5. **10.15 optimises ~2% of GRPO runtime.** The 98% is `sample_group`, which the review
   never addresses.
6. **A4 (bits per character) is not comparable across corpora.** Shakespeare bpc and
   TinyStories bpc cannot go in the same table as an architecture claim.
7. The review says no checkpoints are in the repo. **They are all present locally**,
   gitignored; only GitHub lacks them.

---

## Session log

| Date | WP | What happened |
|------|----|---------------|
| 2026-09-24 | WP5 | **Corpus cleaning built into the notebook and emitted into `bpe.py`** (`6d8ea54`, `9d735c7`). Yoav's steer: **this is teaching material, not a pipeline step** — register is a workshop notebook, so the cleanup is derived and explained rather than hidden in a WP-T script. New Step 4 shows why characters are the *floor* of the vocabulary (`merges = vocab_size − characters − specials`, so a character occurring once costs what `e` costs). `clean_corpus` drops whole documents rather than editing text — an edit you cannot detect beats a loss you can is the same principle as 7.10's raise — and returns the inventory for `bpe_train(chars=...)`. **WP6 must call the identical function** on the pretraining corpus. Committed tokenizer: **79 characters + 944 merges + 1 special**, 109 of 27,630 documents dropped (0.39%), held-out **2.11×**. The teaching payoff is the two-corpus contrast: valid **+9 merges for 0.39%** of documents, train **+124 for 0.014%** — 100× the text gives 2.6× the characters, nearly all junk, so cleaning pays *more* on bigger corpora while costing *less*. **Third prose-ahead-of-measurement correction in this package**: cleaning drops `é` (4 occurrences) and `ñ` (1), so accented Spanish now **refuses** where it encoded before — the OOV section was rebuilt around it, and the lesson improved (cleaning replaced an *accidental* boundary with a *stated* one; both arbitrary, only one writable down and changeable on purpose). `bpe_encode`'s error text also wrongly said such characters "never occurred in the training corpus" and now names both causes. Separator cost re-measured under the cleaned vocabulary: 2.1089× → **2.1088×**. Numbers in `runs.md` T4. |
| 2026-09-24 | WP5/WP-T | **`<|endoftext|>` added, and the corpus policy decided** (`537c83b`). Yoav: **BPE trains on ~10% of the train split, nanochat pretrains on the full train split** — the valid split is ruled out for pretraining (~17 epochs = memorisation). That plan needs one fix, now shipped: `bpe_train(chars=...)` overrides the character inventory, so merges are learned from the sample while the character set comes from a full pass. **Measured, and the reason it is not optional:** the full 2.23 GB train split has **228 distinct characters**, a 10% sample sees **158**, and the **70** it misses cover **296 occurrences out of 2.23 billion** — negligible by frequency, fatal under raise-on-unknown. Merge statistics converge on a sample; character inventories cannot. **Two of the plan's own premises about 8.11 were wrong**: the vocabulary does *not* grow to 1025 (the token is reserved out of the 1024 budget — 88 chars + 935 merges + 1 special at id 1023, so embedding/head shapes are unchanged and 8.11 forces no retrain by itself), and the `\S+|\s+` segmenter does *not* shred the literal (`<|endoftext|>` has no whitespace, so `\S+` takes all of it — the demo cell prints this). The real argument is that a segment is not a token. Cost of the separator: one merge, 2.1065× → 2.1061×. **Left open for WP-T:** 228 characters is 22% of the vocabulary and 137 of them occur <100 times each, so cleaning the corpus instead would buy back ~140 merges — worth measuring before paying for the 1–3 h encode. Train split now on disk; disk at **97%, 34 GB free**. |
| 2026-09-24 | WP5 | **`bpe-tokenizer.ipynb` rebuilt and `bpe.py` generated from it** (branch `wp5-bpe-tokenizer`, `90dc41f` → `320b33e`). Closes 7.1, 7.2, 7.4–7.7, 7.10, A3, A4. **The emission mechanism the plan called undecided is now decided and tested end to end under `nbconvert --execute`**: functions are defined normally, one cell emits them via `inspect.getsource` over the live bindings, so it is idempotent and order-independent where `%%writefile -a` would duplicate or scramble — reuse for `nanochat_model.py` in WP6. Yoav's calls: **raise on unknown characters** (no `<unk>`, so no forced vocab change from this package), **keep generated modules tracked**, **no commit trailers**, review file stays untracked. Default corpus is now the 22.5 MB valid split; compression is measured on 500 held-out stories (**2.11×**, 88 characters + 936 merges, 51 s) and the vocab sweep runs on a fixed 5 MB subsample (49 s, was four full-corpus retrainings). New section derives why the course reports **bits per character** — loss per token is not comparable across tokenizers because the tokenizer picks the denominator — with the corpus caveat attached. **Answered a WP-T question in passing: the shipped tokenizer was trained on the train split** (227 single-char tokens vs 88, **0/1024 ids shared**), so the corpus change forces the retokenise and every existing nanochat/SFT/GRPO checkpoint is now keyed to the wrong tokenizer until WP6–WP8 retrain. Old file preserved as `bpe_tokenizer_trainsplit.pkl`. **Prose falsified by its own run, again**: a draft claimed accented Spanish would be refused; TinyStories contains `é`/`ñ`, so Spanish encodes and *German* refuses — corrected, and the accident-of-the-corpus point is now the section's lesson and Exercise 4. The raise also immediately caught a 2 MB sweep subsample missing `4` and `‘` that the old silent id-0 fallback would have hidden. Figures pinned to `dpi=72` (rcParams are overridden by the inline backend, so dpi must be passed per figure): PNG 88.8k → 57.1k chars, and **the executed notebook was verified to still open for editing** — the WP-N criterion. Numbers in `runs.md` T1. |
| 2026-09-24 | WP4 | **The flush explained, and the fix handed to DataSciPy** (`f753ff4`; issue `yoavram/DataSciPy#15`). Yesterday's rarity explanation was wrong — full house (150 val) and four of a kind (20) are rarer than flush (180) and score 100%. Real cause: **rank alone separates 7 of the 10 classes**, leaving `Nothing|Flush` and `Straight|Straight flush`, so suits are worth ~0.2% of accuracy and the **suit-blind ceiling is 99.82% / 7-of-9 — exactly where the Set Transformer landed**, same unreachable set. The notebook now *computes* that ceiling from the data before any model is mentioned. Two diagnostics added: weight decay drove the transformer's `suit_emb` below its initial scale (15.5x smaller than `rank_emb`), yet a linear probe on the **frozen** representation recovers 66.1% flush recall and 83.76% macro, above the ceiling — the network has the mechanism, keeps the information, and has no reason to use it. **Yoav's scope call: diagnosis stays, fix leaves** — this is a transformers notebook, so resampling/weighting/thresholds go to an FFN-only DataSciPy session (issue #15), keeping execution at ~35 min and one story. Measured for that handoff: balanced sampling takes the transformer to 100%/100% but the FFN to 93.09/79.90, while **√-balanced dominates both metrics for the FFN** (99.14/84.86) and uniform-over-classes *hurts* four of a kind (75→60%). Caveat recorded: straight flush has 14 train / 1 val examples and royal flush 8 / 0, so "100% macro" is partly unfalsifiable. Also corrected pre-commit: a draft generalised "suit embeddings shrank below init" to all three models, which the FFN contradicts (0.2262, above init, and still 0% flush). Numbers in `runs.md` S2. |
| 2026-09-21 | WP4 | **`sets.ipynb` delivered on branch `sets-revision`** (`c827495` WIP, `109d6e4` executed). All of 2.1–2.10 closed. Set Transformer 77.8% macro = **exactly 7/9**: 100% recall on all seven classes it reaches, 0% on the two suit-defined ones, the same figure in three independent runs across two devices. **Yoav's steer mid-package — this is pedagogical material, not research** — rebuilt the Discussion around the mechanism (*how does each architecture compute "do cards i and j share a rank?"*) and cut the statistical caveats to a sentence each; **this steer applies to WP5–WP10 too**. Three findings the review did not anticipate: 2.6 is described backwards in the review (the defect is a **missing** `\\`, not a broken one); 2.3's stronger per-hand permutation test **agrees** with the weak one (1.00% vs 1.10%) rather than exposing more, so a planned exercise resting on the opposite expectation was replaced with an attention ablation; and 2.8's reversed `split` is **not cosmetic** — it shifted the key stream for two models, so every number moved and prose numbers had to come from the notebook's own run. **The plan's CPU-vs-GPU explanation for Deep Sets not reproducing is wrong** — two further CPU runs gave 85.1% and 77.6% against a committed 92.1%; Yoav's mistuned-LR reading fits far better and is now exercise 6. **`sets.ipynb` hit the WP-N edit wall mid-package** at 29,577 tokens, having been comfortably under it when I checked an hour earlier — recovered by cutting printed training logs 200 → 20 lines per model. Corrected before committing: a draft claimed the FFN "fades as hands get rarer", which its own non-monotone column refutes (C5/C7 again). |
| 2026-09-21 | — | **`plan.md` reconciled against `git log` — it had drifted eleven commits.** The session log stopped at `57b55df` while `runs.md` stayed current, so the plan still called WP2 and WP3 "not started" after both had been delivered, and still carried the single-seed decision that W4 overturned. Closed WP2 (Yoav, four prose rows struck through rather than finished — BPTT unnamed, ResearchGate links surviving, nanoGPT uncited, GRU LN un-ablated and kept as an open item) and WP3 (done and overtaken: two of its specification rows were *reversed by measurement*, and its conclusion is not the one the review expected — the GRU wins on bpc). Added **WP-N** for the notebook-size problem that has no tool path around it. The C6/C7 prose debt is delegated to its own issue; the LR-sweep handoff is retired and `handoff-lr-sweep.md` deleted. `plan.md` is now tracked in git — it had been untracked all along, which is why none of this drift was visible. **Rule added to the ground rules: reconcile against `git log` at the start of a session, not only at the end.** |
| 2026-09-21 | WP3 | **C6 and C7 harvested; both falsify something** (`4abebda`). C7 extends the depth curve to 8 and 12 blocks and kills "still descending at 6": the transformer bottoms out at depth 6 (2.041), then 2.050 at 8 and 2.056 at 12. Only the 6 → 12 step clears the 0.015 band, so the rise is gentle — but the curve is not descending at the end, and the transformer never takes an outright win over gru-2L's 2.028. **Same class of error as the one corrected under C5: a curve read past its last measured point.** What the extension does establish is stronger than what it cost — rnn-8L is 4.462 (worse than the *bigram* baseline, approaching the uniform ceiling of 6.07) and gru-8L is 3.369 with seeds spread over 2.2 bpc, against which the transformer's 2.041 → 2.056 is a flat line. "Degrades gracefully with depth where recurrence collapses" is the honest claim. C6 refutes its own hypothesis monotonically: the transformer is the **only** model that gets worse with more history (+0.016 at context 256, +0.046 at 512) because the positional table forces `d_model` down from 72 to 64; both recurrent models are best at 256 and get the longer window free. Exercise 4 already warned students to expect this and can now cite it. `depth_table()` picked up all the new points with no code change. **⚠ The notebook prose is not updated for any of this** — blocked on WP-N. |
| 2026-09-21 | WP2 | **Wall-clock caveat added, and yesterday's size diagnosis corrected** (`a97e6d2`). The committed table mixes sources by construction — result files are C4's (A6000), the transformer rows this machine's (A4000) — so rnn depth 1 reads 4.0 min beside a transformer row at 2.9 min and the minutes column cannot be read down. Bits per character survives a machine change; minutes do not. **The edit had to go through the `nbformat` API**: the notebook is 25.5k tokens with every output stripped, past the read limit on its prose alone, so `NotebookEdit` had no read to work from and `Edit` refuses `.ipynb`. Validated against the cell inventory and a re-parse of every code cell, then re-executed clean. **Correcting yesterday's claim that the inline PNG was the cause**: `dpi=72` cut the image 44 → 28 KB and the file 129 → 113 KB while tokens went 28,756 → **29,043**. Bytes and tokens are not proportional across base64. → **WP-N created**. |
| 2026-09-21 | — | **The executed-notebook edit trap recorded in `CLAUDE.md`** (`5dabbc4`). A notebook is editable before a run and locked after one. |
| 2026-09-21 | WP2 | **Notebooks made runnable, provenanced, and honest about the depth curve** (`40bf38d`). The 3-block transformer is 206,635 parameters, +3.32%, so the notebooks' own 3% assertion fired and `text-transformer.ipynb` **could not run to completion** — replaced by a shared `in_budget()` at `BUDGET_TOL = 0.06`, the bound `cluster/charlm_run.py` already used. "The transformer improves monotonically" is false on our own data (depth 3 → 4 is +0.0243, 3σ) and "RNN peaks at 3" is false too (1 → 3 is 0.0122, inside the band — a plateau, not a peak); that paragraph rewritten as a per-architecture reading against the noise band. The depth curve and wall-clock table had been **hard-coded markdown** while 27 per-seed JSONs sat in `checkpoints/charlm-depth/` — `depth_table()` now renders them, and reproduces the markdown to the digit. Cell 46 was retraining the 1-block transformer at three seeds (~3.6 min) to recompute what `repeat_seeds` had already produced. |
| 2026-09-21 | WP3 | **Experiment retargeted: 200k parameters, 30k steps, 3 seeds** (`1fa42d0`). W1 found the 500k budget overfit a 4.6 MB corpus; W2 found 10k steps left every model still improving; W4 got 2.133 then 2.066 from identical code. **This reverses the single-seed decision Yoav confirmed 2026-09-20** for the character experiment — see the Seeds row. Seed sd ≈ 0.008 is now the instrument: a gap under ~0.015 bpc is noise. |
| 2026-09-21 | WP2 | **Transformer conclusion rewritten around cost, not quality** (`e2f3392`, then `0e12db6` reordering it to lead with cost). The GRU wins on bits per character at this budget and the argument does not need it to lose. MAIN: recurrence is sequential, attention is not — Vaswani Table 1 measured, 6 transformer blocks 2.9 min against 6 GRU layers' 30.9. SUPPORTING **and confounded**: only the transformer converts depth into quality — but our recurrent stacks have no residual connections and the transformer does, so this partly measures residual streams. The cost argument survives the confound; the quality argument does not, and the notebook says so in a blockquote under the table rather than burying it. The residual ablation was **deliberately not run** — it is handed to the student twice, with the actual `deep_feed_forward` patch in `GRU.ipynb` exercise 5. |
| 2026-09-21 | WP3 | **Depth sweep, C5** (`581e90e`). Depths 2/4/6 × 3 architectures × 3 seeds. **Depth is trainable only for the transformer**: at depth 6, rnn 3.343 ± **1.275**, gru 2.291 ± 0.109, transformer 2.041 ± **0.004** — a 300× tighter spread. A 6-layer RNN does not underperform, it **fails to train**: one seed landed at 4.77 against a uniform baseline of 6.07. This is the result that carries "transformers replaced RNNs", and it is a *scaling* claim, not a quality one. |
| 2026-09-21 | — | **`runs.md` created** (`c9a21d5`) — every character-model run, the question it answered, and what came back; append-only, superseded numbers kept with a note. It is committed and is now the authoritative experiment log. Cluster scripts landed alongside (`35348e7`). |
| 2026-09-21 | — | **Cluster and workstation environments documented in `CLAUDE.md`** (`626c4db`) — TAU Slurm targets, the exact account string, the GPU pools, and the ~30–60 s cost of every SSH round-trip. **The `NotebookEdit` failure modes that damaged notebooks** recorded the same day (`5f2466b`): a `cell_type` change on a replace produces a structurally invalid cell, and an `insert` silently shifts every later positional index — which had already overwritten a just-inserted markdown section. |
| 2026-09-20 | WP1 | **Capacity and depth controlled; each notebook now builds 1 layer then 3 at the same budget** (`57b55df`). Yoav's structure. Final table: unigram 4.770, bigram 3.581, rnn 2.065 / rnn-3L 2.133, gru 2.024 / gru-3L 2.036, transformer 2.136 / **transformer-3L 1.994**. Depth bought rnn −0.068, gru −0.012, transformer +0.142 — only attention converts depth into quality, and at one layer the GRU is the best model in the table. This resolved the depth confound properly: cutting the transformer to one block was fair but deleted what makes it work; measuring both depths at fixed budget is fair *and* shows the mechanism. Real stacked-layer RNN and GRU implementations written (the transformer needed none — it already looped over blocks, which is itself part of the argument). `text-transformer.ipynb` gains a **"Why transformers replaced RNNs"** section built on these numbers, leading with the GRU's win and labelling the scale claim an extrapolation; exercises rewritten. Wall-clock is now a table column. ~~Next session: exercise 2, the per-architecture LR sweep — see `handoff-lr-sweep.md`.~~ **That handoff was never actioned and was retired 2026-09-21; the file is deleted.** The depth and context sweeps (C5–C7) went ahead of it. |
| 2026-09-20 | WP1 | **All three review blockers fixed and verified** (`3796b89`). This pulled **A2 and the rest of A3 forward from WP2** — scoring a model through `evaluate_bpc` requires batched forwards, so `jax.lax.scan` over time + `jax.vmap` over batch landed here. `state_policy='none'` is now structural (`h0` built inside `batched_logits`), the three models are actually scored and write their rows, and training really is batch-64. Also fixed: unstable `log(softmax)` in RNN/GRU, checkpoint namespacing under `checkpoints/charlm/`, the GRU multi-layer indirection, train/val curves, the transformer sampler's per-length recompile, and `index.ipynb`'s stale table. **The full comparison now exists:** unigram 4.770, bigram 3.581, rnn 2.078, gru 2.051, transformer 1.952 bpc — but the ordering also tracks parameter count, so it does not isolate architecture from capacity. The namespacing caught a live instance of its own finding: the first fixed RNN run scored the *legacy* checkpoint (2.415) rather than the model it had just trained (2.078). S2/S3 items remain open. |
| 2026-09-20 | WP1 | **WP1 reviewed by subagent — status downgraded to PARTIAL.** Two acceptance items had been ticked in error: no neural model is ever scored (`write_result` is called only for the baselines, so the rnn/gru/transformer rows can never exist), and `state_policy: 'none'` is declared in the config and prose while the RNN and GRU training loops still carry `h` between steps — confound A1, still live, now asserted as fixed in a machine-readable file. Also: the committed config describes a batch-64 experiment the batch-1 code does not run; the numeric-sort fix in `efc767f` makes the legacy 4.8M-step checkpoints win over any new 10k run; RNN/GRU use unstable `log(softmax)` and spike to 13.4 nats at the new LR; lr 3e-3 measured fine at B=64 but the RNN lands below the bigram baseline at B=1. Full findings in the WP1 section. **WP2 must clear the S1 items before WP3 spends GPU time.** |
| 2026-09-20 | WP1 | **WP1 complete** on branch `wp1-shared-experiment`. The experiment is now data: `RNN.ipynb` writes `charlm_split.npz` + `charlm_config.json`; GRU and text-transformer load and hash-verify them, failing loudly if absent. One `evaluate_bpc` entry point for baselines and models alike — the first draft scored the bigram by streaming while models saw windows, which is the A1 confound in miniature; fixed. Baselines committed: Shakespeare 4.770 / 3.581 bpc, TinyStories 4.446 / 3.292 bpc, evaluator checked against log2(V). `char-model-comparison.py` deleted. **Caught a blocker no package listed:** all three char notebooks declared non-existent kernels (`pixi-default`, `conda-env-scipy-py`) and could not start at all; normalised to `python3`. Notebooks deliberately left unexecuted — see the ⚠ note in WP1; **WP3 commits the outputs**. |
| 2026-09-20 | WP-C | **`nanochat_chat.py` → `nanochat-chat.ipynb`** (Yoav), placed between GRPO and the agent notebook; script deleted, `index.ipynb` gains an *Inference* section. Executed on GPU, real outputs committed. Found and fixed a genuine bug in the script's generation loop: growing context ⇒ one XLA compile per token, **429 s** for 60 tokens; fixed-shape buffer + hoisted RoPE/mask + jit ⇒ **4.6 s then 0.2 s, ≈3 ms/token**, byte-identical output. Closes the notebook half of 8.12 early. Three-way sentence-count result: pretrained 4/20, SFT 9/20, GRPO 12/20 — monotone but SE ≈ 0.11, stated as such. **Raises WP6's pasted-model count from three to four** and couples the notebook to WP7's chat-format decision. |
| 2026-09-20 | WP0 | **WP0 complete** on branch `wp0-repo-hygiene`, commit `713d089`, not yet merged. All checklist rows closed, all four acceptance criteria verified. `ipywidgets` added to `pixi.toml` (lockfile updated, `pixi install` re-run). `download_data.py` now puts the 2.23 GB train split behind `--train`. **The three character notebooks lost their outputs** — see the ⚠ note in WP0; WP2 must re-run and commit real ones. Two items were explicitly *not* done and reassigned to WP6: `nanochat_chat.py`'s `top_k`/`top_p`-vs-`generate` reconciliation, and its 4-tuple `load_checkpoint`. |
| 2026-09-20 | — | **Balanced-accuracy run done** (18 min GPU). Set Transformer 77.78% vs FFN 72.32% vs Deep Sets 35.80% vs majority 11.11% — the metric change separates the architectures where plain accuracy did not. All three score 0% on flush. Royal flush has 0 val examples. 2.10 closed (UCI URL works); WP0 env item closed (`pixi install` gives working GPU JAX). Caveat: Deep Sets did not reproduce (84.4% vs 92.1% committed, CPU-vs-GPU plateau sensitivity). |
| 2026-09-20 | — | Web checks: `qwen3.5:9b` confirmed published (1.6/11.1 closed, `4b` chosen as low-VRAM fallback); TinyStoriesInstruct located — **repo name has no hyphen**, the review's `TinyStories-Instruct` would 404; valid split is 26.9 MB plain text, no `datasets` dep needed. |
| 2026-09-20 | — | `sets.ipynb`: 2.4 (train on 5 / eval on 7) dropped — the data is five-card hands, so cell 35's "variable-size input ✓" claim gets softened instead. Balanced-accuracy scoring run dispatched; **no sets checkpoints exist**, so it retrains all three, and it doubles as the WP0 environment smoke test and the 2.10 URL check. |
| 2026-09-20 | — | Yoav on `sets.ipynb`: simplest option — no retraining, no capacity sweep. Keep training as is, switch the metric to per-class recall + macro average, and report the step-budget asymmetry rather than equalising it. |
| 2026-09-20 | — | Yoav: the comparison table is cumulative and lives at the end of each character notebook, not in `index.ipynb`; 1.1 resolved by deleting the landing-page table. |
| 2026-09-20 | — | Yoav: one seed confirmed. Each notebook keeps its own model **and its own harness**; both `charlm.py` and `char-model-comparison.py` are deleted. The experiment is shared as *data* (split `.npz` + config JSON written by `RNN.ipynb`, result JSONs read by `index.ipynb`). |
| 2026-09-20 | — | Feasibility audit by subagent: added WP-T (token pipeline), a compute budget table, and the "Corrections to the review" section. Key catches: no Python env exists (WP0 first item), pin cuda12 not cuda13 (driver 550), `%%writefile` does not execute so the module-generation mechanism is undecided, `char-model-comparison.py` duplicates the char models (since superseded — both `.py` files are now deleted), `sample_group` is 98% of GRPO cost, `uint16` tokens needed to avoid OOM, and 5.2 is overstated. |
| 2026-09-20 | — | Coverage audit by subagent: no finding ID dropped; fixed three WP6 gaps (A3 train-split download, B5 fixed budget, B6 consequence), four narrowed summaries (2.3, 7.10, 8.12, 9.9), the `train_tokens.npy` move-vs-delete conflict, WP2/WP5 header drift, and WP6's dependency on WP5. |
| 2026-09-20 | — | Review read, Part E answered, `plan.md` created. Environment verified: 2× A4000, `qwen3.5:9b` present, all checkpoints on disk. Follow-ups: single seed for all GPU runs; students do not get `.pkl` files (B6 dropped); no delivery date. Code reuse settled: teaching notebooks *generate* `bpe.py` and `nanochat_model.py`, downstream notebooks import them. |
