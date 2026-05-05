# Archived 2026-05-05

This change has been **superseded** by [`gemma-only-execution-plan`](../../changes/gemma-only-execution-plan/), which consolidates the science requirements of this change with the execution-model requirements of `autonomous-agent-pivot`, and applies the Qwen3.5-35B-A3B → published Gemma 4 E4B variants weight-diff swap.

## Why archived (not deleted)
- M0 commits (13a711b) and M1 commits (cc8b7ec) referenced this change's requirements while it was active. Preserving the directory keeps those commits' references resolvable.
- Project history: this is the original 4-person plan that the autonomous-agent pivot, then this consolidation, both built on top of.

## Where things went
- Capability specs (`benchmark-evaluation`, `activation-analysis`, `abliteration-engine`, `weight-diff-analysis`, `research-paper`) → carried forward in `gemma-only-execution-plan/specs/`. The `weight-diff-analysis` spec was substantively rewritten (MoE expert analysis dropped; comparative-Gemma cross-method comparison added).
- Tasks (`tasks.md` sections 1–8) → carried forward in `gemma-only-execution-plan/tasks.md` sections 3–10 (M2a–M5), with the M0/M1 setup sections marked complete with their commit hashes.

## Do not modify
This directory is frozen. To make further changes to the project plan, create a new openspec change that supersedes `gemma-only-execution-plan`.
