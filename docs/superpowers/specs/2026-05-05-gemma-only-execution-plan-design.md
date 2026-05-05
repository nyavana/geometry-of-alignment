# Design: Gemma-only execution plan

**Date:** 2026-05-05
**Status:** Approved (brainstorming complete; ready for implementation-plan generation)
**Supersedes for execution purposes:** `openspec/changes/alignment-geometry-study/`, `openspec/changes/autonomous-agent-pivot/`

## Why

Two pressures converged at the M1→M2 boundary:

1. **Qwen3.5-35B-A3B is a poor fit for the actual hardware.** The original plan put MoE weight-diff analysis on CPU because the 35B-A3B doesn't fit on the 16 GB RTX 4070 Ti Super. CPU iteration over a ~70 GB MoE checkpoint is slow enough that the M3 phase risked bottlenecking the project, and the MoE-expert analysis question — while novel — sits orthogonally to the rest of the project's narrative (everything else is Gemma 4 E4B).

2. **The two-change openspec structure is starting to create friction.** `alignment-geometry-study` (the science) and `autonomous-agent-pivot` (the execution model) overlap in tasks.md sections, so any agent working on M2/M3 has to consult both documents to know what to do. M0 and M1 are complete; the rest of the project will benefit from a single live source of truth.

This design folds both concerns into one consolidation: drop the Qwen phase, swap in a comparative weight-diff analysis across three published Gemma 4 E4B uncensored variants, and collapse the two existing changes into a single forward-looking change.

The science motivation that the swap is *better* than the original plan, not just easier: recent (2025–2026) research has documented that standard Arditi-style abliteration **doesn't work cleanly on Gemma 4** because of two architectural quirks — four RMSNorm layers per decoder block, and shared K/V tensors across layers 24–41. Multiple published "uncensored" Gemma 4 E4B variants exist that solved this in different ways. Comparing their weight diffs against each other and against this project's own M2c abliteration produces a quantitative cross-method comparison on the same parameter space — a result that the original Gemma→Qwen jump could only do qualitatively.

## What changes (high-level)

| Original plan (what's being replaced) | New plan |
|---|---|
| M3: Qwen3.5-35B-A3B MoE weight diff (CPU, ~70 GB checkpoint, slow) | M3: Comparative weight diff across published Gemma 4 E4B abliterations (CPU, ~17 GB each, fast) |
| Section 7 paper: "MoE expert routing and safety" | Section 7 paper: "Why standard abliteration fails on Gemma 4 and how published variants solve it" |
| Cross-reference of refusal directions to weight-diff singular vectors: qualitative, across different models | Cross-reference: quantitative cosine similarity, same parameter space |
| Two openspec changes (`alignment-geometry-study` + `autonomous-agent-pivot`) | One openspec change (`gemma-only-execution-plan`) covering M2 onward; predecessors archived |

Sections 1–6 of the paper (motivation, related work, benchmark, mechanistic analysis, abliteration & selective safety) are unchanged. Course connections to over-parametrization, matrix perturbation theory, and NTK are unchanged.

## Decisions

### D1: Drop Qwen3.5-35B-A3B from the project entirely

**Choice:** The Qwen MoE weight-diff phase is removed. No Qwen download, no MoE expert analysis, no router-modification report. `src/weight_diff/moe_expert_analysis.py` is deleted.

**Rationale:** CPU-only iteration over a 35B-A3B checkpoint is slow enough that completing M3 was a credible risk. The MoE story was the only piece of the project that didn't fit on the GPU and the only piece that used a model other than Gemma 4 E4B. Dropping it makes the project end-to-end consistent on a single model.

**Alternative considered — keep Qwen as a sidebar:** rejected. Half a page of MoE results is not worth the disk space, slow iteration, and execution-model complexity. If the MoE angle is interesting enough to revisit later, it can be its own follow-up project.

**What we lose:** the ability to answer "is safety encoded in expert routing?" That was a real, novel question. The project gives up that contribution in exchange for end-to-end coherence on Gemma 4.

### D2: Three published Gemma 4 E4B variants for the comparative weight diff

**Choice:**
- **OBLITERATUS/gemma-4-E4B-it-OBLITERATED** — primary weight-diff target. Has bf16 safetensors (~17 GB, 7 shards) and well-documented method ("whitened SVD + attention head surgery + winsorized activations," 21 of 42 layers modified, explicit handling of the shared-K/V quirk).
- **TrevorJS/gemma-4-E4B-it-uncensored** (the bf16 source repo, NOT the GGUF-only sibling) — secondary weight-diff target. Different method ("norm-preserving biprojected abliteration") with a public source-code repo.
- **HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive** — benchmark-eval-only third comparison point. GGUF-only, so no weight diff possible.

Plus the project's own M2c abliteration of the same base, used as the fourth comparison point in benchmark eval and the *primary* counterpart in the cross-reference of weight-diff singular vectors against M2b refusal directions.

**Fallback rule:** if TrevorJS turns out to have shape/key incompatibility with the base (different tokenizer config, missing keys, renamed parameters), drop it from the weight-diff portion and proceed with OBLITERATUS only. The Section 7 narrative still works with one method ("OBLITERATUS's surgical modification compared against our textbook abliteration"), it's just less of a comparison.

**Rationale:** N=1 is an anecdote. With two independently-published methods producing weight diffs on the same parameter space, the project can quantify whether different abliteration techniques converge on the same singular vectors or encode safety in different subspaces — either result is paper-worthy. The marginal cost of the second method is small (same code path, ~17 GB more disk).

**What we don't include:** more than three published variants. Diminishing returns; three is enough to make a comparative claim, more is just rows in a table.

### D3: Consolidate the two existing openspec changes into one new change

**Choice:** Create `openspec/changes/gemma-only-execution-plan/` as the single source of truth for M2 onward. Archive both `alignment-geometry-study` and `autonomous-agent-pivot`. Update `CLAUDE.md` and the pointer at the top of `docs/project_plan.md` to redirect agents to the new change.

**Rationale:** M0 and M1 are complete. The two existing changes were never meant to be kept live indefinitely — `autonomous-agent-pivot/tasks.md` task 11.4 already commits to archiving both at project end. The Gemma swap touches both changes (specs in `alignment-geometry-study`, M3 tasks in `autonomous-agent-pivot`), so a surgical third change would force agents to read three documents instead of two. Consolidation pays the writing cost once instead of twice.

**Content carried forward (from `alignment-geometry-study`):**
- `benchmark-evaluation/spec.md` — model list updated (drop Qwen, add three Gemma variants)
- `activation-analysis/spec.md` — unchanged
- `abliteration-engine/spec.md` — unchanged
- `weight-diff-analysis/spec.md` — rewritten (see Spec deltas below)
- `research-paper/spec.md` — Section 7 outline rewritten

**Content carried forward (from `autonomous-agent-pivot`):**
- Worktree topology (six worktrees, but `agent/weight-diff` now does Gemma comparative work)
- GPU lock via flock
- Dispatch contract
- Milestone gates M2/M3/M4/M5 (M0/M1 dropped — already done)
- Human verification gate and STATUS_FOR_HUMAN.md template (with M3 checks updated)

### D4: Naming and traceability

**Choice:** Name the new change `gemma-only-execution-plan`. Record in its `proposal.md` that it supersedes both predecessors and reference the commit hashes that completed M0 and M1. The predecessors are archived but not deleted, so commit history remains traceable.

## File-level impact

### Create
- `openspec/changes/gemma-only-execution-plan/proposal.md`
- `openspec/changes/gemma-only-execution-plan/design.md` (the inline equivalent of this doc, scoped to the openspec format)
- `openspec/changes/gemma-only-execution-plan/tasks.md`
- `openspec/changes/gemma-only-execution-plan/specs/benchmark-evaluation/spec.md`
- `openspec/changes/gemma-only-execution-plan/specs/activation-analysis/spec.md`
- `openspec/changes/gemma-only-execution-plan/specs/abliteration-engine/spec.md`
- `openspec/changes/gemma-only-execution-plan/specs/weight-diff-analysis/spec.md` (rewritten — see Spec deltas)
- `openspec/changes/gemma-only-execution-plan/specs/autonomous-execution/spec.md`
- `openspec/changes/gemma-only-execution-plan/specs/research-paper/spec.md`

### Edit in place
- `CLAUDE.md` — drop the "Qwen3.5-35B-A3B (MoE, CPU-only weight diff analysis)" line in Project Overview; remove the Qwen weight-diff example in the "Running Modules" section; reframe Hardware Constraints (RAM is now used for variant safetensors, not a 35B model)
- `docs/project_plan.md` — update the redirect pointer at top to `openspec/changes/gemma-only-execution-plan/`; rewrite the weight-diff section
- `docs/project_proposal.md` — Models table (Section 4): drop Qwen rows, add three Gemma variants; Section 5.5 description; Section 6 Person D weight-diff scope; Section 7 deliverables list

### Archive
- `openspec/changes/alignment-geometry-study/` → archived (per openspec tooling convention; predecessor is preserved as historical context)
- `openspec/changes/autonomous-agent-pivot/` → archived

### Delete
- `src/weight_diff/moe_expert_analysis.py` — no MoE analysis remaining

### Verify-only (no changes expected, but smoke-test before full sweep)
- `src/weight_diff/compute_diff.py`
- `src/weight_diff/svd_analysis.py`

### Untouched
- `src/benchmark/`, `src/mechanistic/`, `src/abliterate/`
- `data/benchmark_prompts.json` (frozen at M1)
- `scripts/gpu_lock.sh`, `scripts/build_benchmark.py`
- `m1-benchmark-frozen` git tag
- All M0/M1 commits

## Spec deltas (summary)

### `weight-diff-analysis/spec.md` — rewritten

**Removed:**
- `Requirement: MoE expert-level analysis` (and all four scenarios under it: expert modification heatmap, router modification detection, component-type summary, expert/router/shared-expert breakdown)

**Reworded:**
- `Requirement: Low rank indicates abliteration` — now reads as a per-variant claim with a cross-method comparison scenario added.

**Added:**
- `Requirement: Cross-method comparison of published Gemma 4 E4B uncensoring`
  - **Scenario:** When weight diffs from two or more published variants exist, the system SHALL produce a layer-overlay Frobenius bar chart and a singular-vector cosine-similarity table comparing the methods.
- `Requirement: Cross-reference weight-diff singular vectors with M2b refusal directions (quantitative)`
  - **Scenario:** When refusal_directions.pt from M2b and weight-diff SVD outputs from M3 both exist, the system SHALL compute per-layer cosine similarity between the top-k weight-diff left singular vectors and the per-layer refusal direction, and produce a single CSV + figure summarizing the result.
- `Requirement: Architectural quirk handling`
  - **Scenario:** When computing diffs on Gemma 4 E4B, the system SHALL identify shared K/V tensors (layers 24–41 reference layer 24's k_proj/v_proj per published model card analysis) and report each unique tensor's diff exactly once to avoid double-counting.

### `benchmark-evaluation/spec.md` — model list edit

Replace any Requirement-internal reference to Qwen3.5-35B-A3B with the four Gemma 4 E4B targets: base, OBLITERATUS, TrevorJS-bf16, HauhauCS-GGUF. The pipeline already supports both transformers and llama.cpp backends, so the GGUF-only HauhauCS slots in naturally.

### `research-paper/spec.md` — Section 7 outline edit

Section 7 outline shifts from "Weight Diff Analysis (MoE)" to "Comparative weight diff across published Gemma 4 E4B abliterations." Required figures change accordingly:
- `weight_diff_per_layer_overlay.png` (Frobenius per layer, both methods overlaid)
- `singular_value_spectra_per_method.png`
- `cross_method_cosine_table.md`
- `refusal_direction_vs_singular_vector.png`

Sections 1–6 and 8–9 specs are unchanged.

## Tasks (M2–M5 outline, to be expanded in `tasks.md`)

The following is the structural outline. The full `tasks.md` will follow the autonomous-agent-pivot dispatch contract (worktree, branch, GPU policy, commit/push protocol, stop condition per section).

- **M2a — Benchmark evaluation:** as before, but model list is base + OBLITERATUS + TrevorJS-bf16 + HauhauCS-GGUF + the project's own M2c abliteration (when ready). Five rows in the heatmap.
- **M2b — Mechanistic analysis:** unchanged.
- **M2c — Abliteration & selective safety:** unchanged. Note expectation: per recent findings, our textbook rank-1 abliteration may underperform on Gemma 4 due to RMSNorm and shared K/V. This is itself a result for Section 6.
- **M2c-followup — Benchmark on abliterated models:** unchanged in shape, but now includes our own abliteration alongside the three published variants.
- **M3 — Comparative weight diff (replacing Qwen MoE):**
  1. Download safetensors of OBLITERATUS and TrevorJS-bf16; download GGUF of HauhauCS for benchmark.
  2. Pre-flight: assert state-dict keys and shapes match base for each safetensors variant. If TrevorJS fails, log and proceed with OBLITERATUS-only (D2 fallback).
  3. Compute per-parameter Frobenius / relative change / max-abs-change for each (base × variant).
  4. SVD on significantly-modified 2D tensors; effective rank at 95/99%; save top-5 singular vectors per modified weight.
  5. Cross-method comparison: layer-overlay Frobenius bar chart; singular-vector cosine table.
  6. Cross-reference with M2b refusal directions: per-layer cosine between top weight-diff singular vectors and the per-layer refusal direction.
  7. Document each variant's handling of Gemma 4 architectural quirks.
  8. Produce paper Section 7 figures.
- **M4 — Human verification gate:** STATUS_FOR_HUMAN.md template updated. Drop the MoE / router checks. Add: "verify each variant's refusal rate is < 30% on `should_refuse`," "verify the cross-method cosine table exists," "verify the refusal-direction × singular-vector figure exists."
- **M5 — Paper + slides:** unchanged in shape; Section 7 content shifts to the comparative weight-diff narrative.

## Risks / fallbacks

| Risk | Likelihood | Mitigation |
|---|---|---|
| TrevorJS or OBLITERATUS has tokenizer/config mismatch with base | Medium | Pre-flight assert; D2 fallback to OBLITERATUS-only |
| Disk budget — 25 GB existing + 2 × 17 GB ≈ 59 GB just for models | Low | Confirm available disk before download; HauhauCS GGUF is ~5 GB; document in tasks.md as a precondition check |
| Shared K/V double-counting in Frobenius layer plots | Medium | Architectural-quirk Requirement explicitly handles unique-tensor counting |
| All three diffs look identical (rank-1) | Medium | Still paper-worthy ("rank-1 abliteration is method-invariant on Gemma 4 E4B"); narrative pivots to convergence |
| All three diffs look very different | Medium | Also paper-worthy ("safety subspace is under-determined; different methods find different solutions"); ties back to Section 5 |
| Our own M2c abliteration fails on Gemma 4 due to RMSNorm/shared-K/V quirks | Medium-high | This is *itself* a result. Document the failure mode and reference OBLITERATUS's surgical fix as the published workaround. |
| Up-front consolidation cost (more writing this session) | Certain | Front-loaded once; saves repeated cross-doc lookups for every subsequent agent dispatch |

## Resolved decisions (post-review)

1. **M0/M1 inclusion:** include M0 and M1 task lists in the new change's `tasks.md`, with each item marked completed and annotated with its commit hash (e.g. `[x] 1.8 ... *(commit 13a711b on main)*`). They serve as reference for agents reading the new change cold; the task list reads as historical context, not work to do.
2. **Worktree naming:** keep `agent/weight-diff` and the `../gb-wdiff/` worktree path. The work it does has changed; the name still describes the workstream accurately enough.
3. **License audit:** explicit sub-task in M3 step 1 — verify each variant's license card before committing any derivative artifacts (paper figures, JSONs, etc.). OBLITERATUS card already confirms Apache 2.0 inheritance; TrevorJS and HauhauCS to be checked at download time.
