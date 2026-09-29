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

## START HERE — state as of 2026-09-29

**Done, committed and pushed:** WP0, WP1, WP2, WP3, WP4, WP5, WP-T, WP6, WP7, **WP8**,
**WP-J**. Branch `revision2026` is in sync with origin at `996861b`. Nothing is waiting to
go out and no background job is running.

`nanochat-grpo.ipynb` is the current state of the art in this repo and the one to read first
if you are picking this up cold: it trains twice (a words-only reward that gets hacked, then
a judge-based reward that mostly fixes it) and takes **~85 min**. Results: `runs.md` G1
(words-only, as shipped), G2 (the fluency term that failed), G3/G4 (judge probes), G5 (the
shipped judge run).

**Before trusting any number in it:** the notebook has been executed **seven** times.
`242e275` + `4c54a35` were WP8's first pass; three adversarial review rounds forced
re-executions (the clipping metric was read before the update, so it was 0 by construction;
the matcher paid for `carrier`/`bed`/`being`; `per_token_nll` scored an empty completion as
perfectly fluent); then WP-J added the judge. **Every re-execution retrains the policy and
changes both the numbers and the sample text**, which made the prose stale four separate
times — three of them caught by review rather than by me. The shipped numbers are run 7's.
If you re-execute, re-derive every quoted completion and every number in the markdown from
the new output before committing.

**Open, in the order they probably want doing:**

| package | state | note |
|---|---|---|
| **WP-C** `nanochat-chat.ipynb` | ⚠ **reopened** | WP7 invalidated five cells; WP8 has now also changed which GRPO checkpoint exists |
| WP-N | ⬜ not started | only `text-transformer.ipynb` is genuinely stuck |
| WP9 `minisweagent.ipynb` | ⬜ not started | independent of the nanochat chain |
| WP10 `index.ipynb` rewrite | ⬜ do last | WP7 refreshed its SFT line; WP8's GRPO line now needs the same |
| WP-R checkpoint release | ⬜ deferred | near delivery |

**Resolved 2026-09-27:** `checkpoints/nanochat_grpo_best.pkl` has been **deleted** (Yoav's
call). It was pre-WP6 and provably invalid — **797 merges against the current 919**, i.e. a
different tokenizer, so its token ids meant different things — and WP8 no longer produces a
"best" checkpoint at all. **`nanochat-chat.ipynb` still loads that path and will now fail
loudly**, which is the intended outcome: WP-C must point at
`checkpoints/nanochat_grpo_checkpoint.pkl`.

**One thing left as-is, by decision:**

1. **`runs.md` has two `S1` entries** — line ~416 (`sets.ipynb`, WP4) and line ~1194 (SFT,
   WP7). Pre-existing; not renumbered, because renumbering would invalidate citations
   elsewhere. Cite "S1" only with its subject attached.

**WP8's headline, because it changes what the notebook teaches:** GRPO took held-out
constraint satisfaction from **0.459 to 0.923** — and the ground-truth ceiling is **0.820**.
The policy beat the data by using the three required words 8.4 times in total per
completion (vs 4.4 in real stories), welding it to a noun and making it the subject, and inserting it where it
makes no sense; the forgetting check shows language-modelling loss *worse than the
pretrained model*. **This is a genuine reward
hack and it was kept, not retuned** — it is a far better lesson than a tidy 5% gain, and the
notebook's §10 is built on it. Full numbers and the three named strategies: `runs.md` G1.

**Branch `revision2026`: WP8 is committed and pushed.**

**Artifacts on disk** (none in git; all gitignored):

| file | what it is | cost to rebuild |
|---|---|---|
| `checkpoints/nanochat_best.pkl` | pretrained, val 0.8126 @ 63,000 | ~2 h (`nanochat.ipynb`) |
| `checkpoints/nanochat_sft_best.pkl` | **SFT policy, step 1,000, val 0.6533 — GRPO starts here** | ~12 min |
| `checkpoints/nanochat_sft_checkpoint.pkl` | SFT last step 1,986, val 0.6639, *deliberately overfit* | same run |
| `checkpoints/nanochat_grpo_checkpoint.pkl` | **the words-only GRPO policy, step 150 — the reward-hacked one, see WP-R** | ~33 min |
| `checkpoints/nanochat_grpo_judge_checkpoint.pkl` | **the judge-trained policy, step 150 — the better model of the two** | ~35 min |
| `checkpoints/sft_examples.npz` | 11,620 encoded SFT examples; last 1,024 held out, used by WP7 *and* WP8 | 6.7 min |
| `checkpoints/bpe_tokenizer_train.pkl` + `train_tokens.npy` | tokenizer + 1.04B-token corpus | ~16 min |
| `data/TinyStories-Instruct-valid.txt` | 26.9 MB | `python download_data.py` |
| `checkpoints/sft_budget_probe.json` | **committed** — the S2 budget experiment's raw curve | 15 min |

**Untracked files that are not mine to commit, left alone:** `nanochat-review.md` (Yoav's
call: the review stays untracked), `.codex`, `slurm-download-data.sh`.

**Environment:** `~/.pixi/bin/pixi run …`; `TF_CPP_MIN_LOG_LEVEL=3`, `PYTHONPATH=.` for
scripts importing `nanochat_model` or `bpe`.

**Three process lessons from WP8, all folded into the rules below:**
- **Wait on a PID, never on a `pgrep` pattern, and get the PID right.** Two distinct traps,
  both hit repeatedly this session. (i) `while pgrep -f "X"; do ...` matches *its own* command
  line and never exits. (ii) `pixi run jupyter nbconvert ...` spawns a wrapper whose argv
  contains the same string as the real worker, so `pgrep -f "jupyter-nbconvert" | head -1`
  (or `| tail -1`) returns the wrapper or a transient, and the watcher then reports a live
  run as finished. Four wrong-PID watches this session. The incantation that works:

  ```sh
  PID=$(pgrep -f "envs/default/bin/python.*jupyter-nbconvert" | head -1)
  while kill -0 $PID 2>/dev/null; do sleep 60; done
  ```

  Match the interpreter path, not the tool name, and verify with `pgrep -af` before trusting
  it — an `nbconvert` run only writes its output file at the very end, so "no output yet" and
  "already finished" look identical if you are watching the wrong process. **Five
  misidentifications across this session**, including one immediately after writing this
  rule down, because the pattern then appeared inside the *launching* command's own argv.
  The version that finally stopped failing is not a PID at all: have the job `touch` a
  sentinel file when it exits and wait on the file.
- **A notebook can exceed the `Read` limit with outputs already cleared.** This one is 27.9k
  tokens stripped, so `NotebookEdit` was unavailable for the final edits and the `nbformat`
  route was used, validating outputs byte-identical afterwards.
- **Review during the run, and be willing to kill it.** Two reviews found 26 issues; the run
  was killed at ~10 min instead of completing ~40 min of numbers that would have been thrown
  away.

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
| — | Checkpoint distribution | ~~Students do not get the `.pkl` files.~~ **Reversed by Yoav 2026-09-25: ship the `.pkl` files in a GitHub release.** Students are expected to have a GPU, and are *not* expected to train during class — the intended use is to load a shipped checkpoint in class and train off-class if they want to. This removes the constraint that every notebook be runnable from scratch in class-time, and it removes the argument for shrinking any model to fit a laptop. |
| — | **Real model, not a toy** | **Yoav 2026-09-25.** The register is pedagogical, but the artifact should be a genuine nanoGPT at a genuine scale, not a scaled-down demonstration. Where "runnable in minutes" and "a real model" conflict, **the real model wins** and the shipped checkpoint closes the gap. This is what settled WP6's corpus question: full train split, 26.2 M parameters. |
| — | Delivery date | **None set.** Size training runs for correctness, not for a deadline. |
| — | **Register: teaching, not research** | **Yoav 2026-09-24.** These are workshop notebooks. The main line demonstrates *how a mechanism works*; statistical care stays, but compressed to a sentence, not made the climax. WP4's Discussion was rebuilt on this basis (mechanism first: *how does each architecture compute "do cards i and j share a rank?"*). **Applies to WP5–WP10.** Noise bands, reproducibility and seed variance belong in `runs.md`, not in notebook prose. |
| — | **The reference's fixed `1.2`: closed, not adopted** | **2026-09-27.** `nanochat.ipynb` ships RMS QK-norm *without* the reference's `q, k *= 1.2`. Not an evidence question any more: N5 measured Δ = 0.0060 nats/token = **0.0029 per character on parameter-golf's own axis, at p = 0.15**, failing both halves of the standard the notebook itself cites. The one argument for it — `rms12`'s seed sd of 0.0008 against `rms`'s 0.0047 — rests on three runs and is not actionable. Reversing costs a 2.1 h retrain, WP7/WP8 checkpoint regeneration, and every published number. The deciding principle is pedagogical, not empirical: **this notebook ships what it can justify by measurement, not what the reference happens to do**, and adopting a constant it had just shown it cannot justify would undercut its own new section. The constant is explained in the prose and `rms12` appears as one of the three measured variants, so nothing is hidden. Reopen only if exact fidelity to `karpathy/nanochat` becomes a requirement. |
| — | **Two stories in `nanochat.ipynb`, deliberately** | **Yoav 2026-09-26, overriding "one story per notebook" for this notebook only.** It carries an *algorithmic* story (how a GPT works and is trained) and a *data-science* story (a GPT has many hyperparameters, its sensitivity to them is unexpected, so how do you decide?). The second is not a foreign subject that happens to appear here — it is a property of the object being taught, and the QK-norm work produced the material for it. Delivery is a **demonstration that opens a discussion, not an exercise**: present the measurements, frame the question, and **do not write the discussion** — the conclusions are the classroom's. Elaborations: `karpathy/autoresearch` (an agent running 5-minute nanochat experiments and keeping or discarding on `val_bpb`) and `openai/parameter-golf` (16 MB budget, scored on tokenizer-agnostic bits-per-byte, and a record must beat SOTA by 0.005 nats at p<0.01 — usually 3 runs). Both verified against their primary sources 2026-09-26; both postdate the assistant's knowledge and would have been fabricated if guessed. |
| — | **One story per notebook** | **Yoav 2026-09-24.** When a second, genuinely different lesson turns up inside a notebook, split it out rather than carry both. WP4 found a class-imbalance story inside a transformers notebook: the notebook keeps the *diagnosis* (it explains the transformer's own result) and the *fix* was handed to an FFN-only session in `yoavram/DataSciPy` — **issue #15**, filed with all numbers and code. Test to apply: does this lesson explain the notebook's own subject, or is it a different subject that happens to appear here? |
| — | **Review adversarially before closing a package** | **Established 2026-09-25/26.** WP5 was committed and pushed, then reviewed by two subagents — one on code correctness, one fact-checking every number against the notebook's own outputs. They found **two real bugs and six wrong prose claims** in work I had already declared done. Both reviews also *confirmed* the two claims that mattered most, which is the other half of the value: `_apply_bpe_merges` was proved equivalent to textbook BPE, and the pipeline encoder byte-identical to `bpe_encode`. **Do this before saying a package is finished, not after.** Split the reviewers by dimension (code vs. claims) and tell each to report what it checked and found *correct*, so coverage is visible. |
| — | **Grep for the OLD string, not the new one** | **Established 2026-09-26 (WP6, N6), a package after the rule it strengthens.** I recorded in `runs.md`, in a commit message and in a report to Yoav that the notebook "no longer claims the 1.2 is a mere rescale" — while the claim sat word for word three cells above the section built to disprove it. Confirming the *new* text exists does not show the *old* text is gone, and a notebook large enough to need `nbformat` edits cannot be eyeballed. After changing a claim, grep the repo for a distinctive phrase from the **superseded** version and require zero hits. |
| — | **A correction is not landed until you have grepped for it** | **Established 2026-09-26, the hard way.** On 2026-09-24 I corrected three numbers in the notebook and wrote in `runs.md` T5 that they were fixed "there, in `plan.md`, and here" — while T4's table two sections above still carried the old values, and T3 was entirely pre-correction with no warning. A correction note that was **false about its own document**. Rule: after changing a published number, `grep` the whole repo for the old one, and give superseded sections a banner rather than leaving them to be read as current. |
| — | **Resolve ids by position, not by string** | **Established 2026-09-25.** Both of WP5's confirmed bugs came from looking a token id up by its spelling (`encoder[vocab[a] + vocab[b]]`, and a regex over the vocabulary to find special tokens). Strings collide; positions cannot. The vocabulary already had a `[characters][merges][specials]` layout — making that contract explicit (`vocab_layout`) removed both bugs and a whole class of future ones. **WP6 has the same shape of contract** in the checkpoint pytree and the parameter layout: prefer structural invariants that can be asserted over name lookups that can silently agree. |
| — | **Check arithmetic against the definition, not against the story** | **Established 2026-09-26 (WP6, N4).** The notebook claimed unit-L2 QK-norm keeps every attention weight "within about ±13% of 1/256". The real bound is **+28%/−22%**; ±13% is `e^0.125 − 1`, the deviation from the *geometric mean* of the extremes rather than from uniform. It survived two passes and got an exercise built on it **because the conclusion it supported was true** — a 0.78–1.28× spread really does pin the entropy ratio near 1.00, as the A/B then measured at 0.9995. A number that supports a correct story is the hardest kind of error to see. Both S1 findings across WP6's two reviews were of this shape: wrong in a believable, flattering direction, breaking nothing. Brief a reviewer to re-derive each number from its definition, not to check whether it fits the argument. |
| — | **Review while the long run is still going** | **Established 2026-09-26 (WP6).** The code review was dispatched at the *start* of a 2.3 h retrain rather than after it. It found seven bugs, two of which change the training trajectory — so the run was killed at ~20 minutes and restarted on fixed code, at a cost of 20 minutes instead of 2.3 hours. The code is final the moment the edits land; nothing about reviewing it requires waiting for the run. Corollary: **`nbconvert --inplace` overwrites the notebook when it finishes**, so an edit made during a run is destroyed anyway — there is no version of "fix it while it trains" that works. |
| — | **Prefer a reviewer's diagnosis to its patch** | **Established 2026-09-26 (WP6).** The review correctly found that `chars_per_token` was counting `<|endoftext|>` as 13 characters of story, inflating compression by 1.6% and flattering the published bpc. Its suggested fix — drop the separators from both the character and the token count — was itself subtly wrong, because the model *does* spend bits predicting the separator and those bits must be charged against the real text. The bug report was right and the patch was not. Re-derive the fix; only accept the finding. |
| — | **Rehearse a long run at tiny scale first** | **Established 2026-09-26 (WP6).** Before a 2.3 h `nbconvert --execute`, the whole notebook was run end to end at `n_steps=200` on a throwaway copy. It found nothing — and was still worth its ten minutes, because it is the only way to learn that the `inspect.getsource` module-emission cell works under `nbconvert`, that the post-training cells run in the order they appear, and that no late cell references a name defined below it. A failure in the last cell of a long run costs the whole run. Two caveats learned: a rehearsal **writes to the same checkpoint paths as the real run**, and executing cells by `exec` on their source (rather than through a kernel) breaks `inspect.getsource`, so that one cell can only be tested through a real kernel. |
| — | **Measure the mechanism, not just the outcome** | **Established 2026-09-26 (WP6).** The QK-norm A/B produced a 0.432-nat loss gap, which on its own would only say "the fix helped". The number that *explains* it is the attention entropy ratio: **0.9995 under unit-L2, uniform to within 0.05% in every layer**, exactly as the arithmetic predicted, against 0.668 under RMS. A loss gap is evidence; a mechanism measurement is an explanation, and it is what turns a fix into teaching material. Cheap to add to any A/B — instrument the quantity your hypothesis is *about*. |
| — | **Ceilings before blame** | Established in WP4, generalisable. Before attributing a model's failure to its architecture, compute what the *task* permits — WP4's suit-blind ceiling (99.82% / 7-of-9) was derived from the data in three lines with no model, and predicted the trained Set Transformer's score exactly. Cheap, and it converts an unexplained blemish into the notebook's strongest claim. Worth asking in WP6–WP8 too. |
| — | **Check the result against the ceiling, not against zero** | **Established 2026-09-27 (WP8), and it is the lesson of the package.** GRPO improved held-out constraint satisfaction from 0.459 to 0.923 — 29 standard errors, 99% of prompts improved, every statistic immaculate — by *beating the ground-truth ceiling of 0.820*, which is only possible if it is doing a different task than the one intended. It was: using the three required words 8.4 times in total per completion against 4.4 in real stories, welding it to a noun and making it the subject, and inserting it where it makes no sense ("the only infant in the famous infant", scoring 1.00). **The entire measurement apparatus (fixed prompt set, shared keys, clustered SE, paired difference) worked perfectly and bought precision about the wrong quantity.** Only two things caught it, neither a statistic about the reward: the ceiling row, and reading the output. Compute what the task permits *before* training, put it in the same table as the result, and read it last. Generalises WP4's "ceilings before blame". |
| — | **A failed run can be the better artifact — decide on teaching value, not tidiness** | **Yoav's register applied 2026-09-27 (WP8).** The reward hack above could have been tuned away (raise β, stop at step 25) for a modest, respectable improvement. It was kept, and the notebook's discussion was rebuilt around it, because a *measured* reward-hacking event with the instruments that catch it teaches more than a clean 5% gain — and because it vindicates the notebook's own opening claim that RL is good at producing numbers that go up for the wrong reasons. The test is not "does this look like a success" but "does the reader learn more from it". Reopening requires arguing the tidy result teaches more. |
| — | **Pick a reward signal that is not maximised by the failure you are trying to prevent** | **Established 2026-09-28 (WP8 follow-up), and it is the most transferable thing the package produced.** Two candidate fixes for G1's reward hack were measured. A *fluency* term from the frozen pretrained model **failed completely** (G2): likelihood asks "is this text probable", repetition is probable, so the gate sat at 0.99 and never engaged — and the calibration says so in 20 seconds without training, because the model scores its own word-stuffed text (0.71 nats/token) as *more likely* than human stories (0.95). An **LLM judge** asks "is this a coherent story", which the hack cannot satisfy by construction, and it separates cleanly (G3). The rule generalises past this notebook: before building a reward term, ask whether the failure mode you are targeting *increases* the signal you chose. |
| — | **The judge is a specification too, and that is the exercise** | **Yoav 2026-09-28, correcting register.** The instinct after G3 was to ablate the judge prompt across variants and report robustness — research apparatus. This is a workshop: keep the prompt simple and instructive and let the students change it, which teaches "you get what you asked for" one level up. Three prompt decisions, all Yoav's and all improvements: **the judge does not see the required words** (`words_present` already checks those, and telling the judge makes the two instruments correlated by construction rather than independent); **one question, not three** (a compound coherence/grammar/naturalness rating collapsed into one digit hides a weighting nobody chose); and **neutral anchors** — "1 (no) to 5 (yes)", not "1 = word salad", which tells a judge grading word salad what to say. Measured: the blinded prompt separates *better*, lifting human stories from 4,3 to 5,5. |

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

- **One branch for the whole revision: `revision2026`** (Yoav, 2026-09-25 — replaces
  "one work package per branch", which stacked). Finding IDs go in commit messages,
  and the commit is the unit that records a package, not the branch.
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
- **After correcting a published number, grep the repo for the old one** and banner any
  section that is now superseded. See the cross-cutting lesson above — a correction note
  in `runs.md` once asserted a fix that had not been applied to the same file.
- **Have a package reviewed adversarially before closing it**, split by dimension, and
  require the reviewers to state what they checked and found correct as well as what
  they found wrong.
- **Both files are committed, and they divide by *kind*, not by whether they are tracked**
  (Yoav 2026-09-28, correcting an earlier line here that said `plan.md` was not committed —
  it always has been). **`plan.md` holds the plan**: what was done, what is left, decisions
  and reversals. **`runs.md` documents jobs** — running and finished — why each was run and
  what came back, append-only, superseded numbers kept with a note. Do not duplicate a
  results table across both.
- **Neither file is student-facing, and no notebook may reference either** (Yoav
  2026-09-28). They are development artefacts. A notebook that says "see `runs.md` C5" is
  pointing a student at something they do not have and should not need; state the number, or
  say "a cluster sweep not run here", and stop. **Six notebooks currently violate this** —
  see the cross-cutting note in WP-N.

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
| WP7 SFT | **~19 min total, measured (S1)**: 6.7 min BPE-encoding the dataset (cached after, so ~12 min on re-runs), **4.6 min** for 1,986 training steps × 32, 7.9 min for 768 evaluation generations. The old estimate's 407 s per 200 tokens was the *un-jitted* `generate`; WP6's fixed-buffer version is ~0.6 s per ~200-token generation. The 15-epoch budget probe (S2) cost a further 15 min. |
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

### Cross-cutting — remove `runs.md` / `plan.md` references from notebooks  ⬜ 2026-09-28

Neither file is student-facing. Audit on 2026-09-28 found references in six notebooks.
Fixed immediately (prose or comment only, committed outputs untouched): `nanochat-grpo.ipynb`,
`nanochat-sft.ipynb`, `nanochat.ipynb`.

**Still outstanding, because the string is printed in committed output and the fix therefore
needs a re-execution** (57 / 78 / 18 min respectively — fold into WP-N rather than paying it
twice):

| notebook | where |
|---|---|
| `RNN.ipynb` | cell 13 prose; cell 54 prose **and** a printed `'(cluster job; see runs.md C5).'` |
| `GRU.ipynb` | cell 5 prose; cell 42 prose **and** the same printed string |
| `text-transformer.ipynb` | cell 5, 44 prose **and** the same printed string |

Replacement wording: name the measurement, not the log — "a 27-run cluster sweep, not run in
this notebook" — so the sentence stands on its own for a reader who has only the notebook.

---

### WP-N — Get the character notebooks back under the edit limit  ⬜ not started
Not in the review; created 2026-09-21 by a tooling constraint that now blocks WP2.

> **Scope corrected 2026-09-27 (WP7).** The constraint is on a notebook's **source**, not on
> its executed size, and this section's title has misled at least one session into trying to
> shrink the wrong thing. **Every notebook in this repo exceeds 25k tokens when executed** —
> `nanochat.ipynb` 152k, `sets.ipynb` 72k, `bpe-tokenizer.ipynb` 36k, `RNN.ipynb` 31k,
> `text-transformer.ipynb` 26k, `GRU.ipynb` 26k, `nanochat-sft.ipynb` 26k — and that is fine,
> because the route back is always **clear outputs → edit → re-execute**. That route fails
> only when the *stripped* source is itself over the limit, which is exactly and only
> `text-transformer.ipynb` at 25.5k. WP7 spent three executions shrinking a figure before
> checking what the other notebooks actually do; the check takes one command.
>
> Two measured facts for whoever sizes a notebook next:
> - **~3.4 characters per token** for an executed `.ipynb` (measured, not assumed: 88,827
>   chars → 26,126 tokens). Estimate before running: `(source + text output + ~30,000 per
>   figure) / 3.4`.
> - **A figure costs ~9,000 tokens and its cost is dominated by physical dimensions.**
>   Downsampling a 1,962-point line to 199 saved 4 KB of 38 KB; dropping dpi 72 → 68 and
>   0.2 inches of height saved another 4 KB. There is no cheap win — budget for it, or plot
>   nothing. (This supersedes the "trim the figure" reading of the W6 note below, which
>   correctly observed that `dpi` did not help but attributed it to base64 rather than to
>   the fact that pixels are the cost.)

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

### WP-T — Token pipeline  ✅ done 2026-09-25 (`6e89ac3`, `1e9ab00`)
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
      drop any document using anything else, **never edit text**. **Measured on the shipped
      run:** train split 227 distinct → **104 kept**, **230 documents of 2,717,495 dropped
      (0.0085%)**, **+123 merges** (796 → 919); valid split 88 → 79, 109 of 27,630 (0.39%),
      +9 merges. Full numbers in `runs.md` T3/T4/T5.
      ⚠ **Corrected 2026-09-25:** the earlier figures of *389 documents (0.014%)*, *228
      characters* and *+124 merges* were wrong — 389 was measured for the frequency-only
      rule rather than the frequency ∪ ASCII rule actually shipped, and 228 counted the raw
      file including its `<|endoftext|>` literals.

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
      `<`, `|` and `>` never occur in the *validation* split (they do occur in the training split). Cost measured: one merge, held-out
      compression 2.1065× → 2.1061×. **The retokenise is still required**, but by the corpus
      change, not by this row
- [x] **Done 2026-09-25 — `train_tokens.npy` is `uint16`, 2.08 GB against the old file's
      8.5 GB.** Original note: **Store tokens as `uint16`, not `int64`.** Today cell 40 does `jnp.array(train_data)`,
      putting 7.65 GB of train tokens plus 0.85 GB of val on a 16 GB card — around 12–13 GB
      total with params, Adam state and materialised attention weights, against XLA's
      default 75% preallocation of 12.3 GB. A retrain is one batch-size bump from OOM.
      Two-line change, 8.50 GB → 2.1 GB
- [x] **Done 2026-09-25 in 15.9 min** (budgeted 1–3 h) by `build_train_artifacts.py`:
      `checkpoints/bpe_tokenizer_train.pkl` (104 + 919 + 1) and `checkpoints/train_tokens.npy`
      (1,039,345,143 tokens, uint16, 2.08 GB). Verified: separator count equals the document
      count exactly, max id 1023, decodes cleanly across boundaries. Original note:
      Retokenise **once**, producing exactly one new `train_tokens.npy` under `checkpoints/`.
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

### WP6 — `nanochat.ipynb`  ✅ **done and pushed 2026-09-27** — val 0.8126, 0.555 bpc; three reviews; second story added

**Depends on WP5** (it imports the `bpe.py` that `bpe-tokenizer.ipynb` generates), and on
WP-T, which is **done**.

**State at 2026-09-26.** Every source edit landed, the code review (N3) done and its seven
bugs fixed, and the notebook **retrained and re-executed end to end with no errors**:
**best val 0.8126 nats/token** at step 63,000 of 64,000, 126.8 min on one A4000, held-out
**0.555 bpc** against a 3.292 character-bigram baseline, attention entropy ratio **0.687**
(against unit-L2's 0.999). The old model was 1.069. `runs.md` N1–N3 hold the numbers.
**A second story was added after the reviews (Yoav 2026-09-26).** `nanochat.ipynb` now
carries a data-science story alongside the algorithmic one — see the decisions table. It is a
*demonstration that opens a discussion*: two cells after the bits-per-character section
present three QK-norm variants measured under identical conditions, pose five questions, and
**write no conclusions**. Backed by `checkpoints/qknorm_ab.json` (six committed runs, N5) and
pointing at `karpathy/autoresearch` and `openai/parameter-golf`.

The experiment behind it (N5) answered the open `1.2` question and produced something better
than an answer: Δ = 0.0060 nats at p = 0.15, which **clears parameter-golf's 0.005-nat
magnitude bar and fails its p<0.01 evidence bar** — one standard disagreeing with itself. And
a mathematically null code change (× 1.0) moved the result 0.0021, a third of the effect.
**Decision: still ship without the 1.2**, but the prose no longer claims it is a mere rescale
— measured at init, it multiplies the attention-logit std by 1.44 exactly as the algebra says.

**Source size is now the binding constraint: 24,994 tokens, 6 under the limit.** Paid for by
trimming the SFT/GRPO preview (duplicated the notebooks that own it). Any further edit must
cut source first or go through `nbformat`.

**Both review dimensions are done** — N3 (code, 7 bugs) and N4 (numbers, 9 wrong claims).
N4's fixes were markdown-only and went in through the `nbformat` API, because at 539 KB the
notebook is past what `Read` opens and `NotebookEdit` therefore cannot reach it; every code
cell's source and outputs were asserted byte-identical afterwards, so the committed outputs
are still the single clean run.

**The A/B decided it (N1).** Two 4,000-step runs, identical but for the normalisation:

| variant | val @ 4,000 | attention entropy ratio |
|---|---|---|
| **RMS** (the fix) | **1.0707** | 0.668 |
| L2 (old code) | 1.5027 | **0.9995** |

L2's attention is uniform to within **0.05%** in every layer — the arithmetic in 8.1
confirmed by measurement, not by reading the old checkpoint. RMS reaches in 4,000 steps
the loss the old run needed 23,500 for. **E6 confirmed; 8.1 closed.**

- [x] **Corpus/tokenizer coupling asserted.** The notebook no longer encodes anything: it
      loads `train_tokens.npy` and asserts `tokens.max() < vocab_size` and that the
      separator id matches `vocab_layout`, before any training. The "clean at the same
      `min_count`" hazard is gone with the encode itself.
- [x] **8.1 arithmetic** stated in prose; the entropy measurement is in the notebook as
      teaching material, and was run for real in the A/B.
- [x] **The short A/B, run before the retrain** — N1 above.
- [x] **Expected result tempered.** The notebook now says the old model was degenerate,
      not inert: ±13% still carries gradient, the residual stream still carries the current
      token, so it was "current token + almost unweighted bag of context" — which is why it
      still reached 1.069, and why the bug is worth studying precisely because nothing
      crashes.
- [x] ~~Learnable per-head gain?~~ **NO — parameter-free** (Yoav 2026-09-25). Implemented
      as `qk_norm(x) = x / sqrt(mean(x²))`, no learned weight, `1/sqrt(head_dim)` scale
      kept. The reference's fixed `*1.2` is deliberately omitted and the omission is noted
      in prose.
- [x] Replaced the L2 normalisation with RMS; retraining now. **This re-opens WP7 and WP8**,
      which must regenerate the SFT and GRPO checkpoints.
- [x] Attention-map figure across layers (8.10), with the entropy ratio as its quantitative
      half — decoupled from the decision, kept as teaching material.
- [x] RoPE prose now matches the split-half code (8.2), and says the two conventions are a
      fixed permutation apart so weights are not portable between them.
- [x] Scaling table generated (8.3). `param_exact` counts from the shapes, is asserted
      equal to `count_params(init_params(...))`, and reproduces **26,223,104**. The
      embedding + untied head term the formula omits is shown as 4.0% at our scale.
- [x] Vocab size and chars/token read from the loaded tokenizer (8.4): 1024, ~2.14.
- [x] Cell 50's table fixed (8.5): vocab 1024, 26.2M params, `optax.adamw`, plus a QK-norm
      row and the real corpus size.
- [x] "nats per token" throughout; `\text` typo fixed (8.6). bpt vs bpc distinction made
      explicit, since bpc is what the baseline table needs.
- [x] **Re-run end to end (8.7)** — outputs stripped first, every prose/code edit landed
      before execution, then one clean `nbconvert --execute`.
- [x] **Explicit `RESUME`, default off (8.8).** Done first, as the plan required. Resume
      additionally asserts the checkpoint's tokenizer matches the loaded one — the pytree
      shapes are identical across the architecture change, so shape checks cannot catch it.
- [x] WP-T coordination: loads `bpe_tokenizer_train.pkl` (**not** the valid-split teaching
      tokenizer) and the prepared `train_tokens.npy`.
- [x] **A3 fixed, and more cheaply than planned.** The 2.23 GB double-hold is gone with the
      raw-text read. The windows are no longer materialised either: the corpus stays flat
      and `uint16` on the host, `sample_batch` cuts 33 KB batches out of it. The old
      `(4.06M, 257)` device array was 4 GB before the first step.
- [x] **B5:** fixed step budget, no early stopping, best-checkpoint saving retained. The
      budget is *derived* — 20 tokens/param → 64,000 steps — rather than a round number.
- [x] Held-out bpc against the TinyStories character baselines (8.9, A4), on a fixed grid
      of windows so the number is reproducible. The Shakespeare-trained char models are
      deliberately **not** in the table (review correction #6: bpc is only comparable
      within one corpus).
- [x] `generate` hoisted, fixed-shape-buffered and jitted (8.12), ported from WP-C rather
      than re-derived, **plus** `<|endoftext|>` stopping — which the retokenised corpus
      makes possible for the first time and which answers one of `nanochat-chat.ipynb`'s
      own "what is missing" items.
- [x] 8.13 duplicate heading (repurposed, not deleted — it now introduces the
      separator-boundary decode); 8.14 bare `except` gone with the cell it was in;
      8.15 perplexity example now the real one (`log(1024) = 6.93` at init);
      8.16 FLOPs clause added.
- [x] 8.17 vocab-ablation exercise handed to `bpe-tokenizer.ipynb`; 8.18 `scan` exercise
      re-scoped to a throughput measurement, since WP2 already introduced `scan`.
- [x] **`nanochat_model.py` is generated** by the notebook via `inspect.getsource`, same
      mechanism as `bpe.py`, with a reimport self-test (checkpoint round-trip, `generate`,
      and the `‖qk_norm(q)‖ = sqrt(head_dim)` invariant). 17 functions, 360 lines.
      Verified under `nbconvert`: `inspect.getsource` works on a `@jax.jit`-wrapped
      function and keeps the decorator line.
- [x] **WP-C's `top_k`/`top_p` reconciled.** `nanochat.ipynb`'s `sample_token` is now
      WP-C's (greedy at `temperature=0`, `<` not `<=` on the nucleus, returns a JAX array),
      and `generate` takes a checkpoint **dict**. One implementation, in the module.

**Deferred to WP7/WP8/WP-C, deliberately:** switching `nanochat-sft.ipynb`,
`nanochat-grpo.ipynb` and `nanochat-chat.ipynb` from their pasted model copies to
`import nanochat_model`. WP6 *produces* the module; those three notebooks are being
rewritten in their own packages and must regenerate their checkpoints against the new
architecture anyway, so doing the import surgery here would be work done twice.
**Discharged for `nanochat-sft.ipynb` by WP7** (2026-09-27); `nanochat-grpo.ipynb` and
`nanochat-chat.ipynb` still paste the model.

**Code review done (N3), and it paid for itself.** Run *during* the retrain rather than
after, which made the fixes free — the first retrain was killed at ~20 min and restarted on
fixed code, because two fixes change the training trajectory and `nbconvert --inplace` would
have overwritten any edit regardless. Seven bugs, all fixed and re-rehearsed:
`generate` silently returning `''` on a full-length prompt (S2); **`chars_per_token` counting
`<|endoftext|>` as 13 characters of story, inflating compression 1.6% and understating the
published bpc**; a one-short `sample_batch` bound; `step` undefined if the loop never runs;
**resume replaying the data** because the PRNG key reset to 0 while the step counter did not
(now `fold_in(root_key, step)`, which is resumable as well as reproducible);
`os.makedirs('')` on a bare filename in a function that ships downstream; and prose claiming
the char baselines were on "this corpus" when they are on the valid split.

The review also *confirmed* the load-bearing claims: RoPE prose verified element-by-element
against the code, `param_exact` exact for three configs, the padded-buffer `generate`
equal to a full `forward` to 6e-8, no forward references across cells, and no missing name
in the generated module. Detail worth keeping: the reviewer's patch for the bpc bug was
itself slightly wrong (it dropped separators from the token count too, but the model *does*
spend bits on them) — **take a reviewer's diagnosis more readily than its patch.**

**Remaining in WP6:** verify the run's outputs, check source size is still editable (23.5k
tokens of source — under the 25k limit but with little headroom), commit, and run the
**second** review dimension — every number in the committed outputs checked against the
notebook's own claims — before declaring the package done.

### WP7 — `nanochat-sft.ipynb`  ✅ **done and pushed 2026-09-27** (`5045567`) — 0.8437 → 0.6533 nats/token, constraint satisfaction 0.092 → 0.467 against a 0.686 ceiling
Task: **TinyStories-Instruct**, as decided in E3. Numbers in `runs.md` **S1**; the budget
experiment that set the epoch count is **S2**. Reviewed adversarially (below) before commit.

**The notebook was rewritten, not patched**: 20 cells → 28, only four section headings
surviving. The old source was 5.6k tokens of which a third was a pasted model copy; the new
one is 15.0k tokens of which none is.

- [x] **9.1 — the task swap, premise verified rather than assumed.** Prompted with the new
      template, the pretrained model writes dialogue in which **`Words` and `Bad` are
      character names**; it does not read the headers as instructions at all. That
      generation is cell 9, produced *before* training. The old story-continuation task
      could not have produced such a picture, which was the review's central finding.
- [x] **B2/9.3 — the pasted model cell is gone**, replaced by `from nanochat_model import`.
      This also disposes of 9.9 (the `apply_rope(Q,c,s)/norm(Q)` reader trap), which lived
      only in the pasted copy — confirmed by grep, not assumed.
- [x] **9.4 — honest budget.** 6 real epochs over a permutation = 1,986 steps; fixed
      1,024-example validation set scored in full every 100 steps; no early stopping. The
      old run was 80 steps shipped / 160 run, triggered by `patience=4` on a single random
      batch of 8.
- [x] **9.5** output present; **9.6** valid-split choice stated with the V1/V2 caveat;
      **9.7** per-token vs per-example normalisation; **9.8** prompt-mask ablation kept and
      promoted to Exercise 1 ★.
- [x] **B6** — title cell names all four undistributed prerequisites and prices them.
- [x] Two-machines-in-one-notebook fixed by construction: one execution, `JAX 0.9.2 | gpu`.
- [x] `download_data.py`, `README.md` and `index.ipynb` updated.

**The chat-format decision, which WP8 and WP-C inherit.** Template is
`<|endoftext|>Words: …\nFeatures: …\nStory:\n` + response + `<|endoftext|>`, built by
`build_prompt(fields)`. Four choices, each argued in the prose: `<|endoftext|>` as turn
boundary rather than `[INST]` (a reserved id BPE cannot shred, and one the pretrained model
already reads as a boundary — *only possible since WP-T put the separator in the
vocabulary*); `Summary:`/`Random sentence:` dropped (the summary triples the prompt, 35 →
115 tokens of a 256-token window, and both describe the whole story when only its opening
fits); `Words:` kept because it is checkable without a judge; canonical field order so a
later notebook can reconstruct the template exactly.

**Responses are story *openings*.** Median story ~360 tokens against a 256 window, so ~3.5%
fit whole. Truncated at a sentence boundary and terminated with `<|endoftext|>` so the model
learns to stop. Said plainly in the notebook rather than left for the reader to infer.

### What the adversarial review found (two dimensions, both run before commit)

**Numbers/claims dimension — four S1 errors, all mine, all now fixed:**
1. "the response is anywhere from 30 to 230 tokens" — actually **100 to 237, with 90%
   between 186 and 229**. I had read the floor off `min_response=32`. This was load-bearing:
   the per-token-vs-per-example normalisation argument rested on a wide length distribution,
   and the real spread is 1.2×. Rewritten to say the opposite — truncation *homogenises*
   lengths, so the choice barely matters here — which is a better point than the one I had.
2. Three sentences describing GRPO's reward, prompt and starting checkpoint **as fact** when
   the committed GRPO notebook does none of them. Rewritten as a specification, explicitly
   labelled as such.
3. "the committed run trained for 80 steps" — it **ran to 160 of a 200-step budget and
   shipped step 80**. Early stopping cut 20%, not "almost immediately".
4. "about 1% of stories fit" — **3.58%**, ~420 examples. Argument survives, number was 3.5× off.

Plus S2/S3: "three minutes per pass" (it is one), "every downstream notebook imports the
module" (only this one does), a miscounted "jumping four times", median 370 → 357, "eleven
thousand" trained on (10,596), the noise-floor provenance, and two missing prerequisites.

**Code-correctness dimension — no S1 bug; the index arithmetic was verified three ways**
(algebraically, structurally over all 11,620 cached rows, and numerically against a
hand-written loop to 4e-7), including a bit-level test that randomising every padding token
leaves the loss unchanged. It found **one real S2 bug**: Exercise 1 told students to use
`mask = 1 - mask`, which puts weight on the **padding** as well and would spend ~7% of the
objective teaching the model to predict `<|endoftext|>` after `<|endoftext|>`. Fixed with an
explicit prompt mask and a note about why the `-1` is there. Plus S3s: "seen exactly three
times" (4 examples per epoch fall outside the last full batch), an `evaluate` docstring that
overpromised, a misleading `min_response` name, an over-strict vocabulary check that
char-tested `<|endoftext|>`, an `Exercise 6` caveat about the 550 window-filling rows, and a
`setdefault` that recompiled a regex on every call.

**Both reviewers were required to state what they checked and found correct**, and both did
at length. That half is what made the correctness review usable: it tells you which parts do
not need re-checking next time.

### Three results worth carrying into WP8

1. **The two metrics disagree about overfitting.** Loss says the last-step checkpoint is
   clearly worse (+0.0105); constraint satisfaction says it is slightly *better* (0.490 vs
   0.467, ~1.3 SE). Watching only the behavioural number would have called six epochs fine.
2. **`words_present` is stricter than it looks** — `jump` does not match "jumping", which is
   why the ceiling is 0.686 rather than ~0.98. Fair for reporting; **dangerous as a reward**.
3. **The coverage metric wobbles ~0.013 between full-notebook runs** whose computational code
   is byte-identical, while isolated generation is bit-reproducible across processes.
   Mechanism unresolved — see S1, finding 3. WP8 needs a run-to-run floor for its reward.

**Artifacts.** `checkpoints/nanochat_sft_best.pkl` (step 1,000) is **what downstream should
load**; `nanochat_sft_checkpoint.pkl` is the deliberately-overfit last step, kept so the
notebook can measure the difference. `checkpoints/sft_examples.npz` caches the 6.7-minute
encode; `*.npz` is already gitignored.

### Method corrections this package forced

**Rehearse first, execute once.** Every code path on a 900-example subset before the real
run: two minutes, and it confirmed the signal was real before a 20-minute execution.

**Estimate executed notebook size *before* running.** This package burned three executions on
size. `(source chars + text output + ~30,000 per figure) / 3.4` ≈ tokens. And two corrections
to what this plan and `CLAUDE.md` previously implied — see WP-N below.

---

### WP8 — `nanochat-grpo.ipynb`  ✅ **done 2026-09-27** — 0.459 → 0.923 against a 0.820 ceiling: a reward hack, kept deliberately

Full results in `runs.md` **G1**. What this package decided, as distinct from what it measured:

- **The task was replaced, not repaired.** The `[INST]` template and the exactly-N-sentences
  reward are gone. GRPO now optimises `words_present` on SFT's own prompt template, with
  prompts drawn from the 10,596 SFT training examples and evaluation on the same 1,024
  held-out examples SFT used. The two notebooks are now a sequence.
- **Start from `nanochat_sft_best.pkl`** (step 1,000), not the overfit last step.
- **The matcher counts inflections**, not exact spellings — and is deliberately conservative
  about it: no `-er`/`-est`/`-ly`, `-es` gated to sibilants and `-o`, because `let`→"letter"
  and `mat`→"mates" would pay the policy for words it never used. Possessives reduce to their
  stem; hyphenated required words match literally. **The ceiling is re-measured under the
  rule** (0.837/0.622 vs the strict rule's 0.768/0.470) — changing the matcher moves the
  target as well as the score.
- **No early stopping and no "best" checkpoint.** Both selected on noise. WP8 ships exactly
  one artifact, `checkpoints/nanochat_grpo_checkpoint.pkl`, the final policy.
  **Consequence for WP-C:** `nanochat_grpo_best.pkl` is no longer produced, and the copy on
  disk is pre-WP6 and invalid. WP-C must stop loading it.
- **K = 4 inner epochs**, which is the only reason the clipped surrogate does anything: at
  K=1 the clipped fraction is exactly 0 by construction. Measured 0.0094 after all four.
- **The run was kept rather than retuned.** Beating the ceiling by 0.126 is a failure of the
  *specification*, and the notebook now teaches that: §10 names the three strategies the
  policy found, prices the damage with the forgetting check, and says plainly that the
  measurement apparatus bought precision about the wrong quantity. Retuning β or the step
  count to get a modest, tidy improvement would have thrown away the most useful thing the
  package produced. **If anyone reopens this, the burden is to argue the tidy result teaches
  more — not that it looks better.**
- **The reproducibility floor the plan demanded be measured first: it is zero.** Enumerated
  prompts + position-derived keys + one fixed-shape compiled sampler give bit-identical
  per-completion scores within and across processes. A reduced version of that check runs in
  the notebook and asserts, so it cannot silently rot. WP7's unexplained 0.013 drift did not
  reproduce and is not reopened.
- **Sampling was fixed before anything else**, as the plan insisted: ~34× (206 ms per
  194-token generation at B=32, against ~4.4 s). This is what made the package affordable.

**Left undone, deliberately:** the generation budget is one constant (194 tokens) set by the
longest prompt, so 83% of eval prompts have ground-truth stories longer than the budget. Much
better than the old 80, not budget-neutral; the notebook says so rather than claiming it is.

### WP-J — the LLM judge, in the notebook  ✅ **done 2026-09-28** — overshoot +0.102 → +0.008, dead groups 33.7% → 0.0%

Yoav's idea: `minisweagent.ipynb` already depends on a local Ollama model, so a judge is
free here. It is the first thing tried this session that actually addresses G1's reward hack.

**What is measured** (`runs.md` G3, and G4 in flight):

- **As an evaluation column it is decisive.** `words_present` says GRPO beats SFT by
  **+0.521**; the judge says it is **−0.30 ± 0.08 worse** (≈3.8 SE) — the two disagree in
  *sign* and are uncorrelated (r = −0.11). The judge also separates the human stories
  cleanly (84% score ≥4, against GRPO's 10%), which is the sanity check that makes the rest
  readable. Cost **0.5 s per judgement**, so 384 judgements ≈ 3 min.
- **In-loop is viable, and my objections to it were wrong.** I argued cost, dependency and
  reproducibility; all three were weak (parallelism cuts cost ~8×; the repo already depends
  on Ollama; and §6's bit-reproducibility is a claim about the *evaluation*, which stays
  `words_present`). The real risk was granularity — a 1–5 integer judge might give a group
  of G=8 completions identical scores and produce no gradient. **Measured: it does not.**
  Judge dead-group rate ≈25%, comparable to `words_present`'s, and critically the judge is
  *live on groups where `words_present` has saturated and gone dead* — it supplies gradient
  exactly where the word reward has run out. G4 trains 150 steps on
  `reward = words_present × (judge − 1)/4`, multiplicative so neither factor can buy the
  other.

**Decided by Yoav 2026-09-28: the judge goes in the training loop**, not only in the
evaluation. Shipped as sections 11 (the judge as an instrument) and 12 (the judge as the
reward), with section 13 rewritten as a four-reward synthesis. Results in `runs.md` G5.
My recommendation had been evaluation-only on cost grounds; the in-loop result is better
than the probe suggested (+0.008 against the ceiling, against G4's +0.034) and the arc it
gives the notebook — hacked, wrong fix, detected, mostly fixed — is worth the runtime.

**What it cost:** the notebook goes ~40 → ~85 min, gains an Ollama dependency for §§11–12
(guarded: without it those cells print how to enable them and skip), and its source is now
~27.6k tokens, so `Read`/`NotebookEdit` cannot touch it at all — `nbformat` plus hard
validation from here, and any prose fix that needs re-execution costs 85 minutes.

**Superseded, for the record:**

- [ ] **Evaluation column** — ~3 min, guarded so the cell skips with a clear message when
      Ollama is unreachable, so the notebook still runs standalone. Turns §10's "the only
      instruments that saw it were the ceiling and your own eyes" into a measured third
      column. **Low risk, high value; this is the one I would do.**
- [ ] **In-loop reward** — depends on G4. If it lands at or below the 0.820 ceiling with
      required-word mentions falling toward ground truth's 4.4, it is the first reward in
      this series that works, and that is a much better ending than three failures. Costs
      ~35 min of runtime and makes Ollama a hard dependency for *training*, not just for one
      column.
- [ ] Either way: **changing the judge prompt is an exercise**, not an ablation table.
- [ ] Whatever is adopted, the notebook must say that a judge is a proxy too. 9B judging 26M
      is hard to game *because of the capability gap*, which is precisely the protection that
      disappears as the policy gets stronger — cite Gao et al. (2022) on reward-model
      over-optimisation, already in `runs.md` G3.

**Operational, verified 2026-09-28:** `qwen3.5:9b` is pulled and Ollama 0.17.4 is serving on
`localhost:11434`. It is a **thinking** model — a naive call returns an *empty* `response`
with the whole `num_predict` budget spent on hidden reasoning. Pass `"think": false`. A
failed judge call must not silently become a reward value (an early draft returned `1`, a
real score, on timeout); return `None` and substitute the group mean so it contributes no
advantage, and report the count.

---

### WP-C — `nanochat-chat.ipynb`  ⚠ **reopened 2026-09-27 by WP7, widened by WP8**

**WP8 adds two more breakages on top of WP7's five.** (a) `nanochat_grpo_best.pkl` is no
longer produced by `nanochat-grpo.ipynb` and the copy on disk is pre-WP6 — wrong tokenizer,
wrong attention. WP-C loads it. Point it at `nanochat_grpo_checkpoint.pkl`. (b) The GRPO
model's *behaviour* has changed completely: it no longer counts sentences, it stuffs required
words, and §8's discussion of what GRPO bought will read as fiction against the new
checkpoint. WP-C also still carries the repository's last hand-written copy of the model —
`nanochat.ipynb` generates `nanochat_model.py` and every other notebook now imports it.

<!-- original WP-C notes follow -->
### WP-C (WP7 notes)  (was ✅ done 2026-09-20, branch `wp-c-chat-notebook`)

> **Do not treat this package as finished.** WP7 changed the chat template, the reward and
> the SFT checkpoint name, and this notebook is stale in five code cells (9, 14, 16, 18, 22
> hard-code `[INST] … [/INST]`), in cell 0's prerequisites table (`nanochat_sft_checkpoint.pkl`
> is now the deliberately-overfit last step; it wants `nanochat_sft_best.pkl`), and in cell
> 15's prose, which explains a convention the SFT model no longer uses. Its §8 "what is
> missing" entry about `[INST]` being shredded by BPE is now **fixed upstream** rather than a
> known weakness, so that section needs rewriting rather than patching. It also still pastes
> the model instead of importing `nanochat_model`. The notebook must be re-executed after.
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
- [ ] **WP7 consequence — now settled, and this notebook is stale.** WP7 moved SFT to
      `<|endoftext|>Words: …\nFeatures: …\nStory:\n` (built by `build_prompt`), **not** the
      `Features:/Words:/Summary:/Story:` shape guessed here: `Summary:` and
      `Random sentence:` are dropped, the field order is canonical, and the turn boundary is
      the reserved separator rather than `[INST]`. Cells 9, 14, 16, 18 and 22 hard-code
      `[INST]`, cell 0's table names `nanochat_sft_checkpoint.pkl` (should be
      `nanochat_sft_best.pkl`), and cell 15's prose explains a convention the SFT model no
      longer uses. **All of them must change and the notebook must be re-run.** The
      sentence-count measurement has no counterpart in the new SFT task — replace it with
      the word-constraint metric, or drop it
- [ ] **WP6/WP-T consequence:** section 8 states there is no `<|endoftext|>` token and that
      this is why samples end mid-word. If 8.11 lands, that paragraph and the truncation
      caveat in section 5 both need rewriting
- [ ] Exercise 4 asks the student to implement a KV cache. That is the deferred 6.5 / 8.12
      remainder — if a later package implements one, reconcile so the exercise is not
      solved in the repository

---

### WP-R — Checkpoint release and fetcher  ⬜ deferred, do near delivery
**Yoav 2026-09-25: mark it, do it later.** Created because B6 reversed — the `.pkl` files
now ship in a GitHub release — and that touches four notebooks, so doing it once here beats
doing it four times.

Deliberately deferred: the checkpoints it would publish do not exist in their final form
yet. WP6 retrains nanochat under parameter-free QK-norm and the new train-split tokenizer,
and WP7/WP8 regenerate SFT and GRPO on top of that. **Cutting a release before those land
would publish artifacts that are wrong.** Do this once WP6–WP8 have settled.

**The GRPO checkpoint needs a health warning in the release notes, and this is the note.**
`checkpoints/nanochat_grpo_checkpoint.pkl` (step 150, written by WP8's run 6) is the
**reward-hacked** policy: it scores 0.923 on word-constraint satisfaction against a 0.820
ceiling by stuffing the required words, and its held-out language-modelling loss is **0.8388
— worse than the *pretrained* model's 0.8340**, let alone SFT's 0.8247. That is the right
artifact to ship *with `nanochat-grpo.ipynb`*, because the notebook's entire lesson is that
failure and a reader needs the checkpoint that produced the numbers in front of them. It is
the **wrong** artifact for anyone who loads a checkpoint expecting "the best nanochat":
its prose is visibly worse than SFT's. Two consequences for the release:

- Say so on the release page and in whatever the fetcher prints, in one line:
  *"the GRPO checkpoint is deliberately the reward-hacked policy from §9–§10; for the best
  conversational model use `nanochat_sft_best.pkl`."*
- **Ship `nanochat_sft_best.pkl`, not just `nanochat_sft_checkpoint.pkl`.** The `_best` file
  (step 1,000, val 0.6533) is what GRPO starts from and is the best chat model in the series;
  `_checkpoint.pkl` is SFT's deliberately-overfit last step (1,986, val 0.6639), kept only so
  the notebook can measure best-vs-overfit. The candidate list below names the wrong one.

Also note the GRPO pickle carries `best_step`/`best_val`/`train_losses`/`val_log` inherited
from the SFT run it was initialised from — those describe **SFT's** history, not GRPO's.
Harmless, but do not surface them as GRPO metrics in any release tooling. There is no
`nanochat_grpo_best.pkl` any more, by decision (see WP8): selecting the best of six
estimates each carrying ±0.007 is selecting noise.

- [ ] Decide the release tag and what goes in it. Current candidates, ~830 MB total:
      `bpe_tokenizer_train.pkl`, `nanochat_checkpoint.pkl`, **`nanochat_sft_best.pkl`** (not
      `_checkpoint.pkl` — see the warning above), `nanochat_grpo_checkpoint.pkl` **and
      `nanochat_grpo_judge_checkpoint.pkl`**, which WP-J added: the judge-trained policy is
      the better of the two GRPO models (0.828 words found against the 0.820 ceiling, judge
      3.66 against 2.34) and the hacked one is only worth shipping because §§9–13 are about
      it. Label them so nobody guesses. **`train_tokens.npy` is 2.08 GB — decide whether it
      ships or is rebuilt by `build_train_artifacts.py`** (16 min, needs the 2.23 GB
      download; rebuilding is probably the better trade)
- [ ] A fetcher: a `download_data.py --checkpoints` flag or a small `fetch_checkpoints`
      helper. One implementation, imported by every notebook that needs it
- [ ] Every notebook that needs a checkpoint **fetches it rather than assuming a previous
      notebook produced it**. This is the actual behaviour change, and it affects
      `nanochat.ipynb` (resume/load), `nanochat-sft.ipynb`, `nanochat-grpo.ipynb` and
      `nanochat-chat.ipynb`
- [ ] State in each notebook which artifacts it needs and where they come from, so a
      student who has not run the previous notebook is not left guessing
- [ ] Checksums, so a truncated download fails loudly rather than unpickling into nonsense
- [ ] Revisit `.gitignore`: the checkpoints stay untracked, but the release URLs and tag
      should be recorded somewhere version-controlled

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

## Branch state

**One branch for the whole revision: `revision2026`.** Yoav's decision 2026-09-25 —
stacking a branch per work package meant each new branch carried the previous one's
unmerged commits, and the PR-per-package rule could only be honoured by merging them in
order. All revision work now continues on `revision2026`, which was created at `33c74c0`
and therefore already contains WP4, WP5 and WP-T's decisions.

Superseded branches, kept for now and safe to delete whenever: `sets-revision` (local
only, WP4), `wp5-bpe-tokenizer` (local and pushed, WP5 + WP-T). Their commits are all
contained in `revision2026`, so nothing is lost by deleting them.

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

- [x] **Checkpoint distribution (B6) — resolved 2026-09-25: ship the `.pkl` files in a
      GitHub release.** Students have GPUs and are not expected to train in class.
      **The release itself is deferred — see WP-R below.**
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
| 2026-09-27 | WP7 | **`nanochat-sft.ipynb` rewritten on TinyStories-Instruct, reviewed, and re-run; numbers in `runs.md` S1/S2.** 20 cells → 28. The review's central finding (9.1) was that the old story-continuation task *could not demonstrate anything*; the replacement's premise was therefore **verified before any code was written** — prompted with the new template, the pretrained model writes dialogue in which `Words` and `Bad` are character names. **Held-out response loss 0.8437 → 0.6533** (0.1903, vs a 0.005 noise floor) and **constraint satisfaction 0.092 → 0.467 against a ground-truth ceiling of 0.686**, on a fixed 1,024-example validation set — replacing an old result measured on *one random batch of 8*. **The chat format is decided and WP8/WP-C inherit it**: `<|endoftext|>` as turn boundary rather than `[INST]`, `Words:`+`Features:` kept, canonical order, `build_prompt` as the single source. **Yoav asked whether a longer SFT would help; a 15-epoch probe (S2) says no** — validation bottoms at 3.0 epochs and rises after, while constraint satisfaction stays flat inside its error bars. The budget still went 3 → 6 epochs, not to train better but so the committed plot *shows* the turn: that makes best-checkpoint saving visibly necessary and yields the best-vs-overfit comparison. **Adversarially reviewed on two dimensions before commit**, which paid: four S1 prose errors of mine (response lengths 100–237 not 30–230, which forced rewriting the normalisation argument; GRPO claims stated as fact when the committed GRPO notebook does none of them; the old run ran to 160 and *shipped* 80; 3.5% of stories fit, not 1%), plus one real bug — Exercise 1's `1 - mask` weights the padding. The correctness reviewer verified the index arithmetic three ways and found no S1. **Three results carried to WP8**: the loss and behavioural metrics *disagree* about overfitting (loss clearly worse, coverage slightly better, ~1.3 SE); `words_present` is stricter than it looks (`jump` ≠ "jumping", hence the 0.686 ceiling) and is dangerous as a reward; and the coverage metric wobbles ~0.013 between full runs whose code is byte-identical, mechanism unresolved. **Cost three executions to notebook-size confusion** — corrected in WP-N and `CLAUDE.md`: the binding limit is on *source*, every notebook here is over when executed, and a figure costs ~9k tokens dominated by physical size. |
| 2026-09-26 | WP6 | **WP6 edits landed and the QK-norm question settled by experiment.** Outputs stripped first (8.7 permits it, and the notebook was 28k tokens — unreadable by `Read`), then every prose and code edit landed *before* execution, per the size ground rule. **The A/B (N1) is the result:** 4,000 steps of RMS QK-norm against 4,000 of the old unit-L2, identical in init, data order and validation batches. Val **1.0707 vs 1.5027**, a 0.432-nat gap visible from step 200. The decisive number is not the loss but the **attention entropy ratio: L2 sits at 0.9995 — uniform to within 0.05% in every one of the 8 layers** — against RMS's 0.668. 8.1's arithmetic confirmed by measurement rather than by reading the old checkpoint, and E6 closed. **Tempering that matters:** RMS reaches in 4,000 steps the 1.069 the old run needed 23,500 for, so the old model was not broken, just paying ~6x the compute; the notebook now says so. Also landed: `RESUME=False` first (the pytrees are identical across the change, so nothing would have errored); A3 fixed more cheaply than planned (corpus stays flat/`uint16` on the host, `sample_batch` cuts 33 KB batches — the old code built a 4 GB device array before step 1); B5 fixed budget *derived* from Chinchilla (64,000 steps); the scaling table now generated and asserted against `init_params` (reproduces 26,223,104); held-out bpc against the TinyStories char baselines; attention maps (8.10); jitted fixed-buffer `generate` ported from WP-C plus `<|endoftext|>` stopping; and **`nanochat_model.py` is now generated** by the notebook with a reimport self-test. **A 200-step rehearsal of the whole notebook** was run before the real one — it found nothing, but it is the only way to confirm `inspect.getsource` survives `nbconvert` without spending 2.3 h to find out. It also overwrote the old checkpoints, which WP6 invalidates by design. Retrain in flight; review and commit still to come. |
| 2026-09-26 | WP5 | **Two subagent reviews of WP5, and the fixes** (`1c5fcff`, `72dc682`). Reviewed *after* the package was committed and pushed — which is the wrong order, and is now a ground rule. **Confirmed sound:** `_apply_bpe_merges` ("apply each rule once in training order") was *proved* and empirically verified equivalent to textbook BPE over 6,148 segments, 0 disagreements; and `build_train_artifacts.py`'s streaming encoder is byte-identical to `bpe_encode` on 300 real documents — so the 1.04 B-token corpus was valid. **Two real bugs, one root cause:** ids were resolved by *string*, so a learned merge spelling a special token stole its id, and a learned token shaped `<|…|>` was silently treated as special (−50% compression on the reviewer's corpus). Fixed structurally by making the `[characters][merges][specials]` layout explicit (`vocab_layout`) and reading ids off positions; `SPECIAL_TOKEN_RE` deleted. **Verified behaviour-preserving — re-encoding reproduces `train_tokens.npy` byte-for-byte**, so no rebuild. Also fixed: `bpe_train` silently returning a short vocabulary (the toy example printed `vocab_size=14` after asking for 15, in committed output I had read twice), the sweep omitting `special_tokens` so its 1024-point was not the shipped tokenizer, a hand-typed module preamble that could drift, and two `build_train_artifacts.py` footguns (bare `KeyError` minutes in; `--sample-frac` silently ignored when a cached tokenizer existed). **Six prose claims were wrong**, including "forty-six times more documents" (98×; I conflated it with the drop-rate ratio *in the sentence that states that ratio correctly*), "`minisweagent.ipynb` imports `bpe.py`" (it does not), and "our longest tokens are bare words" (contradicted by the chart directly beneath). **And `runs.md` T5 claimed corrections had landed that had not** — T4 still carried 228 / 389 / +124. T2/T3 bannered, T4 corrected. Three cross-cutting lessons promoted to the decisions table. |
| 2026-09-25 | WP5/WP-T/WP-R | **Single branch `revision2026`; WP-T delivered; the notebook now samples.** Branch consolidation (Yoav): stacking a branch per package meant each carried the previous one's unmerged commits — `revision2026` cut at `33c74c0` already contains WP4/WP5, and `sets-revision`/`wp5-bpe-tokenizer` are superseded. **WP-T done in 15.9 min** (budgeted 1–3 h): `bpe_tokenizer_train.pkl` (104 chars + 919 merges + 1 special) and `train_tokens.npy` (1,039,345,143 tokens, **uint16, 2.08 GB** against the old int64 file's 8.5 GB), verified by separator count, max id and boundary decoding. **Three published numbers were wrong and the run caught them** — 230 documents dropped (0.0085%) not 389 (0.014%), 227 distinct characters not 228, +123 merges not +124; the 389 had been measured for the frequency-only rule rather than the frequency ∪ ASCII rule shipped. **The notebook now samples** (`SAMPLE_FRAC = 0.10`), closing a preach-vs-practice gap Yoav caught: the sampling section explained a mechanism the code never exercised, which also left `chars=keep_chars` inert. A new cell measures character coverage by sample size (10% sees 72/79, missing `/0678=` and a backtick), and the fraction was chosen by measurement — 10% is within **0.08%** of full-corpus compression for a third of the time. The sampling trap and the always-keep-ASCII clause turn out to defend the same characters. Yoav's decisions recorded: **B6 reversed** (ship `.pkl` in a release; GPUs assumed, no in-class training), **real model not a toy**, **parameter-free RMS QK-norm** (verified against nanochat), **full train split at 26.2 M**. **WP-R created and deferred** — a release cut now would publish checkpoints WP6–WP8 are about to invalidate. Numbers in `runs.md` T5. |
| 2026-09-24 | WP5 | **Corpus cleaning built into the notebook and emitted into `bpe.py`** (`6d8ea54`, `9d735c7`). Yoav's steer: **this is teaching material, not a pipeline step** — register is a workshop notebook, so the cleanup is derived and explained rather than hidden in a WP-T script. New Step 4 shows why characters are the *floor* of the vocabulary (`merges = vocab_size − characters − specials`, so a character occurring once costs what `e` costs). `clean_corpus` drops whole documents rather than editing text — an edit you cannot detect beats a loss you can is the same principle as 7.10's raise — and returns the inventory for `bpe_train(chars=...)`. **WP6 must call the identical function** on the pretraining corpus. Committed tokenizer: **79 characters + 944 merges + 1 special**, 109 of 27,630 documents dropped (0.39%), held-out **2.11×**. The teaching payoff is the two-corpus contrast: valid **+9 merges for 0.39%** of documents, train **+123 for 0.0085%** — 100× the text gives 2.6× the characters, nearly all junk, so cleaning pays *more* on bigger corpora while costing *less*. **Third prose-ahead-of-measurement correction in this package**: cleaning drops `é` (4 occurrences) and `ñ` (1), so accented Spanish now **refuses** where it encoded before — the OOV section was rebuilt around it, and the lesson improved (cleaning replaced an *accidental* boundary with a *stated* one; both arbitrary, only one writable down and changeable on purpose). `bpe_encode`'s error text also wrongly said such characters "never occurred in the training corpus" and now names both causes. Separator cost re-measured under the cleaned vocabulary: 2.1089× → **2.1088×**. Numbers in `runs.md` T4. |
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
