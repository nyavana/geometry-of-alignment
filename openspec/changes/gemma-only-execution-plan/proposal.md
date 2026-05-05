## Why

This change consolidates the two predecessor openspec changes (`alignment-geometry-study`, `autonomous-agent-pivot`) into one source of truth and adjusts the project scope by replacing the Qwen3.5-35B-A3B MoE weight-diff phase with a comparative weight-diff across three published Gemma 4 E4B uncensored variants. Two pressures motivated the consolidation:

1. **The Qwen MoE phase fits poorly on the actual hardware.** The 35B-A3B is CPU-only on this server (no GPU room), and CPU iteration over a ~70 GB MoE checkpoint risks bottlenecking the project. The MoE-expert analysis sits orthogonally to the rest of the project's narrative, which is otherwise entirely Gemma 4 E4B.

2. **Two-change overlap creates dispatch friction.** Agents working on M2/M3 currently must consult both predecessors' tasks.md sections, increasing the chance of cross-document drift. M0 and M1 are complete; the rest of the project benefits from a single live change.

The science motivation that the swap is *better* than the original plan, not just easier: standard Arditi-style abliteration is documented (2025–2026 literature) to fail cleanly on Gemma 4 due to four-RMSNorm-per-block and shared K/V tensors across layers 24–41. Multiple published "uncensored" Gemma 4 E4B variants solved this in different ways. Comparing their weight diffs produces a quantitative cross-method comparison on the same parameter space — which the original Gemma → Qwen jump could only attempt qualitatively.

## What Changes

- **BREAKING (relative to predecessors)** Drop Qwen3.5-35B-A3B from the project entirely. No download, no MoE expert analysis, no router-modification report. `src/weight_diff/moe_expert_analysis.py` is deleted.
- **BREAKING (relative to predecessors)** Replace the MoE weight-diff workstream with a comparative weight-diff across three published Gemma 4 E4B uncensored variants:
  - `OBLITERATUS/gemma-4-E4B-it-OBLITERATED` — primary weight-diff target (bf16 safetensors, ~17 GB; method: whitened SVD + attention head surgery + winsorized activations).
  - `TrevorJS/gemma-4-E4B-it-uncensored` (the bf16 source repo) — secondary weight-diff target (norm-preserving biprojected abliteration; has public source-code repo).
  - `HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive` — benchmark-eval-only third comparison point (GGUF only, no weight-diff possible).
- Add explicit fallback rule: if TrevorJS shape/key-mismatches with base, drop it from weight-diff and proceed with OBLITERATUS only. The Section 7 narrative still works with one method; it is just less of a comparison.
- Replace the `weight-diff-analysis` capability spec deltas: remove `Requirement: MoE expert-level analysis`; add `Requirement: Cross-method comparison of published Gemma 4 E4B uncensoring`, `Requirement: Cross-reference weight-diff singular vectors with M2b refusal directions (quantitative)`, `Requirement: Architectural quirk handling`.
- Reframe Section 7 of the paper from "Weight Diff Analysis (MoE)" to "Comparative weight diff across published Gemma 4 E4B abliterations."
- Carry forward unchanged: `benchmark-evaluation` (model list updated only), `activation-analysis`, `abliteration-engine`, `autonomous-execution`, and `research-paper` (Section 7 outline updated only). Predecessor MODIFIED scenarios from `autonomous-agent-pivot/research-paper` are merged inline into the new `research-paper` spec.
- Archive the two predecessor changes to `openspec/archive/<name>/` (preserves provenance; removes from `openspec list` scope). Each archived change gets a `_NOTE.md` redirect pointing forward to this change.
- Update `CLAUDE.md`, `README.md`, `docs/project_plan.md`, `docs/project_proposal.md` to drop Qwen references and add the three Gemma variants.

## Capabilities

### New Capabilities
- _(none — all capabilities are carried forward from predecessors with the modifications described above)_

### Modified Capabilities
- `weight-diff-analysis`: substantively rewritten — see spec delta. MoE-specific requirements removed; comparative-method, cross-reference, and architectural-quirk requirements added.
- `benchmark-evaluation`: model list updated to drop Qwen3.5-35B-A3B and add three Gemma 4 E4B variants. Pipeline behavior unchanged.
- `research-paper`: Section 7 outline shifts from MoE-focused to comparative-Gemma-abliteration-focused. The autonomous-agent-pivot's MODIFIED scenarios (writeup gated by human verification, all numeric claims traceable) are merged in.
- `activation-analysis`, `abliteration-engine`, `autonomous-execution`: no semantic change.

## Impact

- **Compute**: Same hardware as predecessors. CPU + 100 GB RAM is now used for ~17 GB-per-variant Gemma safetensors weight diff (instead of ~70 GB Qwen MoE). GPU usage unchanged.
- **Models**: dropped — Qwen3.5-35B-A3B (original + uncensored). added — `OBLITERATUS/gemma-4-E4B-it-OBLITERATED` (safetensors + GGUF), `TrevorJS/gemma-4-E4B-it-uncensored` (bf16 safetensors), `HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive` (GGUF). Base `google/gemma-4-E4B-it` and `google/gemma-4-E2B-it` unchanged.
- **Dependencies**: no new deps. `safetensors`, `torch`, `transformers`, `bitsandbytes`, `llama-cpp-python` already in `requirements.txt`.
- **Disk**: +~34 GB for two new safetensors + ~5 GB for HauhauCS GGUF on top of existing ~25 GB. Pre-flight check in M3 step 1 verifies disk budget.
- **Predecessors**: `openspec/changes/alignment-geometry-study/` and `openspec/changes/autonomous-agent-pivot/` move to `openspec/archive/`. Their commit history is preserved; `openspec list` will show only this new change.
- **Source code**: `src/weight_diff/moe_expert_analysis.py` deleted. `compute_diff.py` and `svd_analysis.py` are model-agnostic and unchanged (verified with smoke-test in M3 step 2).
- **Authorization**: Instructor's autonomous-agent authorization (recorded in auto-memory `project_authorization.md`) is unchanged.
- **Risk mitigation**: `Requirement: Architectural quirk handling` makes the shared-K/V double-counting risk an explicit spec scenario.
