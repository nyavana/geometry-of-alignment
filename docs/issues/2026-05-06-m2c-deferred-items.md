# M2c — items deferred to follow-up dispatch

**Date:** 2026-05-06
**Branch:** `agent/abliteration`
**Context:** First M2c dispatch (commits a113fc5, b25c361, 9c5cef6) shipped
sections 5.1 - 5.7, 5.9 and launched the long-running 5.7 sweep ensemble
in the background under `scripts/gpu_lock.sh`. Items below are explicitly
deferred — the GPU is still held by the sweep, so any GPU-bound follow-up
must wait for it to finish (and release the lock) or be queued via
`scripts/gpu_lock.sh`.

## Sweep watchdog

- Sweep launcher PID (parent shell): `451347`
- Sweep flock holder PID (lock cleanup target): `451354`
- Wrapped python PID: `451361`
- Lock holder file: `~/.geometry-of-alignment/.gpu.lock.holder`
- Log: `/tmp/m2c-sweep.log`
- Output: `$RESULTS_DIR/ablation_results/sweep_results.json` (alpha + layer
  + random control merged into a single JSON; the script's existing
  contract).
- Expected runtime: 2 - 4 hours (9 alphas + 9 layer configs + 1 random
  control = 19 iterations × 50 prompts × ~5 s/prompt ≈ 80 - 160 min on
  4070 Ti Super at 8-bit, plus weight-snapshot restore overhead).

The sweep will release the GPU lock on its own when it exits.

## Deferred to follow-up dispatch

### 5.7 sweep result analysis

Once `sweep_results.json` lands, parse it for:
- alpha curve shape (refusal rate vs alpha) — does it monotonically
  decrease, or saturate by alpha=0.7?
- layer-subset comparison — is `peak_band_4_17` (added in this dispatch)
  on par with `all_42`? If so, paper claim "the high-signal band M2b
  identified is sufficient — the rest of the layers carry no extra
  refusal capacity."
- random-direction control — should give refusal rate ≈ base 100% on
  `should_refuse`. If it doesn't, the abliteration mechanism itself is
  noisy.

**Important**: per the M2c 5.6 finding (the abliterated model still
refuses ~92% of `should_refuse` even at alpha=1.0 across all 42 layers),
expect the alpha curve to be *flat* on `should_refuse` and only the
emergency_medical / over-refusal categories to show appreciable
movement. This is the predicted Gemma 4 architectural-quirk failure.

### 5.8 capability preservation (MMLU + GSM8K)

Status: not started. Needs GPU. Follow-up agent should:
- Load `$RESULTS_DIR/models/gemma-4-e4b-abliterated/` and base
  `google/gemma-4-E4B-it` separately (not in parallel — VRAM cap).
- Pull MMLU + GSM8K subsets via `datasets` library, score N=200 each.
- Diff abliterated vs base and write
  `$RESULTS_DIR/ablation_results/capability_preservation.json`.

### 5.10 selective abliteration

Status: design done (see commit 9c5cef6 body and
`$RESULTS_DIR/activations/category_cosine_summary.json`); needs GPU.

The category-direction analysis (5.9) found that
`emergency_medical_vs_should_refuse` cosine ≈ +0.001 in the M2b peak
signal band L4-17. The category directions tensor is at
`$RESULTS_DIR/activations/category_directions.pt` (dict
`{category: {layer_idx: (1, hidden_dim) tensor}}`) — the follow-up
should:

1. Load the abliterated-base model (or re-load fresh base + apply the
   medical-only direction from `category_directions.pt`).
2. Project out *only* the `emergency_medical` direction (and optionally
   wilderness_survival), NOT the should_refuse direction.
3. Evaluate against the full benchmark; expect:
   - over-refusal on `emergency_medical`: target <10%
   - refusal on `should_refuse`: target >80%

The existing `src/abliterate/selective_safety.py` re-extracts
activations on the GPU which is wasteful. Either:
- patch `selective_safety.py` to read `category_directions.pt` directly
  (bypassing the `compute_category_directions` re-extraction step), or
- write a new minimal script in `scripts/` that loads the .pt file and
  calls `abliterate_model` with only the medical sub-dict.

Note: the **partial-removal finding from 5.6** means that even if 5.10
"works" semantically (medical cosine is orthogonal to should_refuse
cosine), the magnitude of effect on emergency_medical may be small —
the same architectural resistance that limited 5.6 to ~8 pp removal on
should_refuse will also limit 5.10's removal of medical over-refusal.
That's still a paper finding.

### 5.11 figures

- `$RESULTS_DIR/figures/alpha_sweep.png` — line plot of refusal rate
  vs alpha, faceted by category if possible.
- `$RESULTS_DIR/figures/layer_subset_comparison.png` — bar chart, one
  bar per LAYER_CONFIGS entry.
- `$RESULTS_DIR/figures/selective_safety_table.md` — markdown table
  with rows = experiments, columns = (refusal rate, over-refusal,
  capability delta).

These should land after 5.7 / 5.10 finish; they read from the JSON
sweep results.

## Section 6 (M2c-followup)

The abliterated model is at:

```
/home/nyavana/columbia/6699/shared/results/agent/abliteration/models/gemma-4-e4b-abliterated/
```

Section 6 (`agent/benchmark-eval` worktree) can run
`evaluate.py --backend transformers --use-8bit --model <path>` against
this directory directly. Reminder: the **classifier-regex fix** in
this dispatch (commit b25c361) is on `agent/abliteration` only — section
6's worktree should rebase before running, otherwise its
`should_refuse` rate will be artificially low (matches the M2c 5.6
finding that the un-fixed classifier missed all "I cannot ___" forms).

## Dispatch contract reference

Per the original dispatch prompt:
- Worktree: `/home/nyavana/columbia/6699/gb-ablit/`
- Branch: `agent/abliteration`
- GPU policy: gpu-lock-required (the sweep currently holds it).
- The orchestrator will dispatch a follow-up agent for 5.8 / 5.10 / 5.11
  once the sweep releases the lock.
