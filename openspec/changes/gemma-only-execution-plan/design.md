## Context

The project's research goal is to locate where refusal/safety behavior lives in the weights of Gemma 4 E4B-it and characterize whether and how it can be selectively perturbed. M0 (environment bootstrap) and M1 (benchmark freeze, 340 prompts + 640 variants) are complete; the `m1-benchmark-frozen` tag on `main` marks the boundary. M2–M5 are the live workstreams: benchmark evaluation, mechanistic analysis, abliteration + selective safety, comparative weight diff, human verification gate, and paper writeup.

The science context that justifies the comparative weight-diff phase: standard Arditi-style abliteration is documented (2025–2026 literature) to fail cleanly on Gemma 4 due to (a) four RMSNorm layers per decoder block (instead of the two found in earlier transformer designs) and (b) shared K/V tensors across layers 24–41. Multiple published Gemma 4 E4B uncensored variants — OBLITERATUS, TrevorJS, HauhauCS — each handle these quirks differently. Comparing their weight diffs on the same parameter space yields a quantitative cross-method comparison and lets the project's own M2b refusal directions be cosine-checked against the published variants' singular vectors.

This change supersedes archived predecessors `alignment-geometry-study` and `autonomous-agent-pivot` (see `openspec/archive/`).

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

### Decision 6: M6 cascade is staged, gated, and bounded — not a parallel "try everything"

**Choice:** Implement the M6 rank-1 abliteration follow-up as a five-stage cascade (Stage 0 → 1 → 1.5 → 2 → 3 → 4) with a three-band gate (≤30% / 30–85% / >85%) at each escalation point. Stages 0–3 run only on the 48-prompt stratified subset (`stratified_50.json`); Stage 1.5 confirms any cracked headline at n=42 (single-category `should_refuse` subset) before declaring a paper-grade result; Stage 4 (full 344-prompt benchmark) runs on the winning variant only and is gated on operator confirmation.

**Rationale:** The five-hypothesis space (H1 bnb int8 edit path, H2 chat-template-derived direction, H3 winsorization, H4 Gram-Schmidt, H5 norm-preserving biprojection) is too large to test in parallel — total cost would be ~9 hours for full-benchmark sweeps even at one variant per hour, and there is no need to run each hypothesis at full resolution if a cheap one cracks. Ordering by (a) plausibility given the published evidence and (b) cost-to-test puts the bnb int8 edit-path test first (zero new code, ~30 min), the direction-quality stack second (one helper script, ~2.5 h), and the algebraic biprojection fix last (~3 h). Each gate either short-circuits the plan with a finding or escalates to the next stage.

**Why the three-band gate, not a binary pass/fail:** at n=6 a single classifier flip moves the rate by 16.7 pp, so a 30–85% result is "significant partial effect, but not enough on its own" — the cheapest hypothesis explained some-but-not-all of the gap, and the next stage tests whether a second ingredient closes the rest. A binary pass/fail would either falsely promote noise or falsely reject partial effects.

**Why Stage 1.5 (n=42 confirmation):** the n=6 smoke is too coarse to publish off; the 42-prompt single-category run is sufficient resolution for the headline claim and far cheaper (~20 min) than the full 344-prompt benchmark (~2.5 h). It is also where hand-audit of "refusal-then-comply" false negatives happens — TrevorJS's model card flagged 3 such cases out of 100 in their own evals.

**Alternative considered — full reimplementation of TrevorJS as Stage 1:** rejected. Spending ~3 h on a faithful reimplementation skips the diagnostic insight from cheaper tests. If H1 (bnb int8 edit path) is the cause, the staged plan finishes in ~4 hours total with a *causally-isolated* headline; the full reimplementation finishes in ~3 hours with a "TrevorJS works, we don't" headline that is strictly less informative.

**Alternative considered — parallel dispatch of all five hypotheses:** rejected on hardware grounds. The 4070 Ti holds one bf16 E4B copy at a time (~10–12 GB peak with KV cache); two simultaneous copies OOM. Sequential dispatch is forced by the single GPU; the cascade gate makes that sequencing diagnostically useful instead of merely costly.

**What we lose:** if multiple hypotheses are jointly necessary (e.g., bnb edit path AND chat-template AND winsorize all matter, none alone), the cascade reports "necessary in combination with prior ingredients" rather than per-ingredient sufficiency. Stage 2.5 (optional unstacked isolation) is the escape hatch — added as an opt-in re-test of the marginal ingredient, not as a default — to convert "necessary" into "sufficient" claims when the operator wants the stronger paper claim.

**Cross-spec note:** the framing-assertion discipline that prevents a positive Stage 0a result from being miscommunicated as "int8 quantization is the cause" is captured operationally in two places — `abliteration-engine/spec.md` (the `bf16 vs bnb int8 edit-path isolation (M6 H1)` requirement, scenario "Framing assertion in run output") and `autonomous-execution/spec.md` (the `M6 staged cascade with three-band gate` requirement, scenario "Framing assertion in commit messages"). HauhauCS's quantized-but-uncensored Q8 GGUF is the standing counterexample that keeps the claim narrow.

## Risks / Trade-offs

- **[TrevorJS or OBLITERATUS shape/config-mismatch with base]** → Pre-flight assert in M3 step 2; D2 fallback to OBLITERATUS-only.
- **[Disk budget — ~34 GB for new safetensors + ~5 GB GGUF]** → Pre-flight `df -h` check in M3 step 1.
- **[Shared K/V double-counting in Frobenius]** → Architectural-quirk spec Requirement makes this an explicit acceptance criterion.
- **[All three diffs look identical]** → Still paper-worthy ("rank-1 abliteration is method-invariant on Gemma 4 E4B"); narrative pivots to convergence.
- **[All three diffs look very different]** → Also paper-worthy ("safety subspace is under-determined"); ties back to mechanistic Section 5.
- **[Our own M2c abliteration fails on Gemma 4 due to RMSNorm/shared-K/V quirks]** → Itself a result. Document the failure mode and reference OBLITERATUS's surgical fix as the published workaround. M6 then runs the staged causal-isolation cascade (Decision 6) to identify which single ingredient closes the gap — a positive M6 result reframes the M2c headline from "rank-1 fails on Gemma 4" to a causally-isolated single-ingredient finding.
- **[M6 Stage 0a positive result is misframed as "int8 quantization is the cause" when the actual claim is narrower]** → Spec acceptance criterion (`abliteration-engine/spec.md`) plus dispatch-time framing assertion: any Stage 0a positive result is reported as "bnb int8 in-place edit path" specifically. HauhauCS's quantized GGUF scoring 0% is a known counterexample to the broader claim.
- **[M6 smoke gates at n=6 are too coarse to support paper claims]** → Stage 1.5 confirmation at n=42 before declaring any "cracked" headline; Stage 4 full-benchmark only after Stage 1.5 confirmation and operator approval.
- **[M6 paper claim is overstated — single ingredient when actually a composition]** → Phrase paper claims as "necessary" not "sufficient" unless Stage 2.5 unstacked isolation confirms single-variable causation. The decision tree in `docs/M6_PROPOSAL_RANK1_FOLLOWUP.md` Section 6 encodes this nuance.

## Provenance

This change consolidates two predecessor openspec changes — `alignment-geometry-study` (the science framing) and `autonomous-agent-pivot` (the autonomous-agent execution model) — both archived under `openspec/archive/<name>/` with `_NOTE.md` redirects pointing here. M0 (environment bootstrap) and M1 (benchmark freeze) were completed under those predecessors and are referenced from `tasks.md` for traceability; their work is not re-executed.

## Open Questions

_All open questions from the brainstorming design doc were resolved before this change was written. Resolutions are folded into the decisions above._
