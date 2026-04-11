## Why

The original `alignment-geometry-study` plan assumed four humans working in parallel on a shared codebase. Two facts changed that assumption:

1. The EECS 6699 course instructor explicitly authorized the team to run a fully autonomous LLM agent workflow for the final project, including long-running Claude Code sessions.
2. The server owner (person operating this machine) is now the sole human in the loop. The three other "people" in the original plan are replaced by Claude Code agents that execute on this box.

The existing plan still describes the *science* correctly (benchmark, activations, abliteration, weight diff, paper), but the *execution model* (GPU schedules by weekday, "Person A → D" section assignments, manual 4-person coordination) is no longer how the work will actually happen. This change updates the execution model without touching the scientific goals.

## What Changes

- **BREAKING** Replace the four-person workload division (Persons A/B/C/D) with a single-server, agent-driven execution model. The original section assignments in `docs/project_plan.md` and the spec deltas that reference "Person X" are retained only as historical context.
- Introduce a git-worktree-per-workstream topology. Each of the four science workstreams (benchmark-evaluation, activation-analysis, abliteration-engine, weight-diff-analysis) is executed inside its own worktree + feature branch so multiple Claude Code agents can run in parallel without stepping on each other's working trees.
- Retire the human GPU-sharing calendar (Mon/Wed/Fri etc.) and replace it with a single-writer GPU lock that agents acquire before any CUDA work. Activation extraction and abliteration share the same 16 GB card and must serialize regardless of which agent is driving.
- Revise the milestone schedule from "4 weeks × 4 people" to agent-schedule milestones: **Milestone 0 (environment)**, **M1 (benchmark frozen)**, **M2 (GPU experiments complete)**, **M3 (CPU weight diff complete)**, **M4 (human verification gate)**, **M5 (paper + slides)**. Wall-clock dates are removed; advancement is gated on artifact completion, not calendar days.
- Add a mandatory **human verification gate** (M4) after all experiments complete and before the paper-writing phase begins. At the gate, agents must produce a `STATUS_FOR_HUMAN.md` document describing: what was run, what results exist, which claims in the plan were confirmed or contradicted, and what actions the human must take (eyeball figures, sanity-check outlier results, rotate any credentials, approve proceeding to paper phase).
- Add a final **writeup phase** (M5) that runs *after* human verification. Paper drafting and slide preparation are explicitly the last pre-submission autonomous work; no new experiments are launched in M5.
- Add an agent-dispatch contract: each agent is spawned with (a) a worktree path, (b) a branch name, (c) a scoped prompt referencing one tasks.md section, (d) an instruction to commit and push its branch on section completion. Agents do not merge; merging is a human decision at the verification gate.

## Capabilities

### New Capabilities
- `autonomous-execution`: Agent-driven execution workflow for this project — worktree topology, branch naming, GPU locking, dispatch contract, commit/push protocol, human verification gate, and the STATUS_FOR_HUMAN handoff artifact.

### Modified Capabilities
- `research-paper`: Paper and slide production moves from "Week 4, written collaboratively by 4 authors" to "M5, drafted by a single agent or small agent team after the human verification gate passes". No scientific content changes.

## Impact

- **Docs**: `docs/project_plan.md` is superseded by this change for the execution-model parts. A short pointer at the top of `project_plan.md` will redirect future readers to `openspec/changes/autonomous-agent-pivot/` for the current workflow. The scientific content of `project_plan.md` (category lists, hook sketches, SVD analyses) remains authoritative.
- **Tasks**: The existing `alignment-geometry-study/tasks.md` is not rewritten. Its science tasks are re-pointed to agent-owned sections via this change's `tasks.md`. References to "Person A/B/C/D" become labels, not ownership.
- **Branches**: New long-lived feature branches created from `main`: `agent/env-bootstrap`, `agent/benchmark-eval`, `agent/mechanistic-analysis`, `agent/abliteration`, `agent/weight-diff`, `agent/writeup`. Each lives in its own worktree under `../geometry-of-alignment-worktrees/<branch-short-name>/`.
- **Compute**: No new hardware. Same GPU (4070 Ti Super 16 GB) + CPU (100 GB DDR4). The only new compute primitive is a filesystem-based lock (`.gpu.lock`) that agents flock() before loading a CUDA model.
- **Dependencies**: No new Python deps. Requires `git worktree` (already present) and `tmux` or Claude Code's background task system for parallel agent sessions.
- **Ethics / authorization**: Instructor has authorized autonomous agent workflow for this project (recorded in auto-memory). No change to the ethics framing of the research itself.
- **Risk**: Autonomous runs can burn GPU hours on broken code. Mitigation: every experimental agent must run a 10-prompt smoke test before committing to a full sweep, and the GPU lock has a 6-hour max-hold with auto-release.
