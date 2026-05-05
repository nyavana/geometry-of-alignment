# Archived 2026-05-05

This change has been **superseded** by [`gemma-only-execution-plan`](../../changes/gemma-only-execution-plan/), which consolidates this change's execution-model requirements with the science requirements of `alignment-geometry-study`, and applies the Qwen3.5-35B-A3B → published Gemma 4 E4B variants weight-diff swap.

## Why archived (not deleted)
- M0 tasks 1.1–1.9 and M1 tasks 2.1–2.6 in this change's `tasks.md` are checked off and reference commits 13a711b, e341ff2, b95480e, cc8b7ec, 40b0fe5, f982302. Preserving the directory keeps those task statuses traceable.

## Where things went
- `autonomous-execution` capability spec → carried forward verbatim in `gemma-only-execution-plan/specs/autonomous-execution/spec.md`.
- `research-paper` MODIFIED scenarios (writeup gated by human verification, all numeric claims traceable) → folded into `gemma-only-execution-plan/specs/research-paper/spec.md` as ADDED scenarios under the consolidated `Requirement: Presentation slides`.
- M0 / M1 / M2 / M3 / M4 / M5 tasks → carried forward in `gemma-only-execution-plan/tasks.md`. M0 and M1 are marked `[x]` with commit hashes for traceability.
- The Qwen3.5-A3B MoE weight-diff workstream is **removed** (not migrated). The replacement is the comparative weight-diff across published Gemma 4 E4B variants in `gemma-only-execution-plan/tasks.md` section 7.

## Do not modify
This directory is frozen. To make further changes to the project plan, create a new openspec change that supersedes `gemma-only-execution-plan`.
