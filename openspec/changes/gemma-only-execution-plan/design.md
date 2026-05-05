## Context

The project entered M2 with two openspec changes already in flight: `alignment-geometry-study` (the science) and `autonomous-agent-pivot` (the execution model). M0 and M1 are complete (`m1-benchmark-frozen` tag on `main`). Two facts changed the plan at the M2 boundary:

1. **Qwen3.5-35B-A3B is a poor fit for this hardware.** The 35B-A3B doesn't fit on the 16 GB GPU, so the original plan put it on CPU with the 100 GB RAM. CPU iteration over a ~70 GB MoE checkpoint is slow enough that completing M3 was a credible risk. The MoE-expert analysis question is novel but sits orthogonally to the rest of the project, which is end-to-end Gemma 4 E4B.

2. **Multiple published Gemma 4 E4B abliterations now exist** (OBLITERATUS, TrevorJS, HauhauCS, and Heretic-derived variants). Recent (2025–2026) literature documents that standard Arditi-style abliteration fails cleanly on Gemma 4 due to (a) four RMSNorm layers per decoder block and (b) shared K/V tensors across layers 24–41. The published variants each solved this differently. Comparing their weight diffs produces a quantitative same-parameter-space comparison.

This change folds both observations into one consolidation.

## Goals / Non-Goals

**Goals:**
- Single openspec change as source of truth for M2 onward.
- Same-model end-to-end story from benchmark (M2a) through mechanistic (M2b) through abliteration (M2c) through comparative weight-diff (M3) — Gemma 4 E4B throughout.
- Quantitative cross-reference between weight-diff singular vectors (M3) and refusal directions (M2b) — possible because both live in the same parameter space.
- Drop the Qwen-MoE-on-CPU bottleneck.

**Non-Goals:**
- Re-running M0 or M1. Their work is complete and committed.
- Redoing the abliteration math. `src/weight_diff/compute_diff.py` and `svd_analysis.py` are already model-agnostic over safetensors.
- Investigating MoE expert routing for safety encoding. That question is dropped, not deferred.
- Building a general framework for comparing N abliteration methods. Three is enough.

## Decisions

### Decision 1: Drop Qwen3.5-35B-A3B from the project entirely

**Choice:** No Qwen download, no MoE-expert analysis, no router-modification report. `src/weight_diff/moe_expert_analysis.py` is deleted.

**Rationale:** CPU iteration over ~70 GB MoE checkpoint is too slow to fit the project timeline. The MoE story was the only piece of the project not on Gemma 4. Dropping it makes the project end-to-end consistent.

**Alternative considered — keep as a sidebar:** rejected. Half a page of MoE results is not worth the disk + slow CPU + execution-model complexity. If MoE turns out to be a follow-up worth doing, it can be its own project.

**What we lose:** the "is safety encoded in expert routing?" question. Acknowledged.

### Decision 2: Three published Gemma 4 E4B variants, with a fallback to one

**Choice:**
- **OBLITERATUS/gemma-4-E4B-it-OBLITERATED** — primary weight-diff target. Has bf16 safetensors (~17 GB, 7 shards). Method documented: whitened SVD + attention head surgery + winsorized activations; 21 of 42 layers modified; explicit handling of the shared-K/V quirk.
- **TrevorJS/gemma-4-E4B-it-uncensored** (bf16 source repo, NOT the GGUF-only sibling) — secondary weight-diff target. Method: norm-preserving biprojected abliteration. Has public source-code repo for sanity-checking.
- **HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive** — benchmark-eval-only (GGUF only).

**Fallback:** if TrevorJS pre-flight (state-dict keys + shapes) fails, drop it and proceed with OBLITERATUS only. Section 7 narrative still works with N=1 (degraded comparison).

**Rationale:** N=1 is an anecdote; N=2 enables cross-method comparison on the same parameter space. The marginal cost of the second method is small (same code path, ~17 GB more disk). HauhauCS adds a third behavioral data point with no weight-diff cost.

**Alternative considered — only OBLITERATUS:** acceptable as the fallback shape but suboptimal as the primary plan; reduces Section 7 from a comparison to a single deep dive.

### Decision 3: Consolidate into one new change; archive predecessors

**Choice:** Create `openspec/changes/gemma-only-execution-plan/` as the single live change for M2 onward. Move `openspec/changes/alignment-geometry-study/` and `openspec/changes/autonomous-agent-pivot/` to `openspec/archive/` with a `_NOTE.md` redirect.

**Rationale:** M0 and M1 done. Predecessors were always meant to be archived at project end (autonomous-agent-pivot/tasks.md task 11.4 commits to this). Adding a third overlay change for the Gemma swap would force agents to consult three documents. Consolidation pays the writing cost once.

**Why archive via `mv` (not `openspec archive`):** the openspec CLI's `archive` command merges a change's ADDED Requirements into `openspec/specs/` (the steady-state library). We are not transitioning to steady state — we are folding two changes into one new change. `mv`-ing to `openspec/archive/` keeps the predecessors out of `openspec list` scope without polluting the main spec library, and the new change can use clean ADDED Requirements (rather than MODIFIED Requirements layered over an artificial steady state).

### Decision 4: Worktree naming and milestone numbering preserved

**Choice:** Keep the six worktrees and their branch names from `autonomous-agent-pivot` (`agent/env-bootstrap`, `agent/benchmark-eval`, `agent/mechanistic-analysis`, `agent/abliteration`, `agent/weight-diff`, `agent/writeup`). Keep milestone names M0–M5. M0 and M1 are referenced in the new tasks.md as completed sections; M2–M5 are open.

**Rationale:** Renaming `agent/weight-diff` to `agent/comparative-diff` would force a worktree recreation and break the GPU-lock assertions that reference the existing branch names. The work it does has changed; the name is still descriptive enough.

### Decision 5: Architectural-quirk handling is a first-class spec requirement

**Choice:** Add `Requirement: Architectural quirk handling` to `weight-diff-analysis/spec.md` that explicitly identifies Gemma 4 shared K/V (layers 24–41 reference layer 24's `k_proj`/`v_proj`) and requires diffs to be reported on unique tensors only (no double-counting).

**Rationale:** The OBLITERATUS model card documented this as a real bug they hit ("applied projection 18× to the same tensor, corrupting it"). Our weight-diff code performs subtraction, not projection, so the failure mode is different — but Frobenius-per-layer plots could double-count modifications if not handled. Surface this in the spec rather than as a comment in code.

## Risks / Trade-offs

- **[TrevorJS or OBLITERATUS shape/config-mismatch with base]** → Pre-flight assert in M3 step 2; D2 fallback to OBLITERATUS-only.
- **[Disk budget — ~34 GB for new safetensors + ~5 GB GGUF]** → Pre-flight `df -h` check in M3 step 1.
- **[Shared K/V double-counting in Frobenius]** → Architectural-quirk spec Requirement makes this an explicit acceptance criterion.
- **[All three diffs look identical]** → Still paper-worthy ("rank-1 abliteration is method-invariant on Gemma 4 E4B"); narrative pivots to convergence.
- **[All three diffs look very different]** → Also paper-worthy ("safety subspace is under-determined"); ties back to mechanistic Section 5.
- **[Our own M2c abliteration fails on Gemma 4 due to RMSNorm/shared-K/V quirks]** → Itself a result. Document the failure mode and reference OBLITERATUS's surgical fix as the published workaround.
- **[Up-front consolidation cost]** → Front-loaded once instead of paying it twice.

## Migration Plan

1. Create new openspec change directory + skeleton (Task 1 of this plan).
2. Write proposal, design, six spec files, tasks.md (Tasks 2–10).
3. Validate with `openspec validate gemma-only-execution-plan` (Task 11).
4. Move predecessors to `openspec/archive/`, add `_NOTE.md` redirects (Tasks 12–13).
5. Update `CLAUDE.md`, `README.md`, `docs/project_plan.md`, `docs/project_proposal.md` (Tasks 14–17).
6. Delete `src/weight_diff/moe_expert_analysis.py` (Task 18).
7. Final verification — `openspec list` shows only the new change; grep for stale Qwen references (Task 19).
8. Single squashed commit per task, all on `main` as a doc-level bootstrap exception (matches M0/M1 precedent).

## Open Questions

_All open questions from the brainstorming design doc were resolved before this change was written. Resolutions are folded into the decisions above._
