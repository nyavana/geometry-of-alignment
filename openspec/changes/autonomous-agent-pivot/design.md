## Context

The `alignment-geometry-study` change already scaffolded the science codebase: `src/benchmark/`, `src/mechanistic/`, `src/abliterate/`, `src/weight_diff/` all have runnable modules, `data/benchmark_prompts.json` exists, and both `gemma-4-E2B-it` and `gemma-4-E4B-it` are downloaded to `model/`. Results directories are empty (only `.gitkeep`) — no experiment has actually been run end-to-end yet. The original tasks.md was written for four humans coordinating on a single main branch.

Two things changed since that plan was written:
1. Instructor authorized a fully autonomous LLM-agent workflow for this final project (recorded in auto-memory: `project_authorization.md`).
2. The three non-operator "people" are gone. The server owner is the only human. Everything the original plan assigned to "Person A/B/C/D" must now be executed by Claude Code agents running on this machine.

The science from the original plan is still correct — what needs to change is the *execution topology*, *coordination protocol*, and *milestone schedule*. The GPU (4070 Ti Super 16 GB) is still the bottleneck, the CPU + 100 GB RAM is still available for the Qwen weight diff, and Gemma 4 E4B at 8-bit is still the mechanistic analysis model.

Constraints:
- **Single-writer GPU**: 16 GB VRAM is not enough to run two Gemma 4 E4B experiments simultaneously. Activation extraction and abliteration sweeps must serialize.
- **Plenty of CPU RAM**: 100 GB is enough to run the Qwen weight diff concurrently with GPU work without any contention.
- **No CI**: there is no build/test pipeline, so agents must run their own smoke tests and self-verify before claiming a task complete.
- **Long running sessions**: Claude Code sessions can run for hours, but individual agent invocations should be scoped to one worktree-branch to keep context focused.

## Goals / Non-Goals

**Goals:**
- Let multiple Claude Code agents work on independent workstreams in parallel without stepping on each other's files or git state.
- Make GPU contention impossible to cause data corruption or OOM crashes, even if two GPU-touching agents run at the same time.
- Produce a single canonical artifact at the human verification gate (`STATUS_FOR_HUMAN.md`) that describes the entire state of experiments, what to eyeball, and what to approve.
- Preserve all the scientific requirements of `alignment-geometry-study` verbatim — this change modifies only execution, not science.
- Keep the writeup phase (paper + slides) strictly *after* human verification so a human has signed off on the raw results before Claude starts quoting them in LaTeX.

**Non-Goals:**
- Automating merges. Agents push branches; the human merges (or asks an agent to merge a specific branch, explicitly, at verification time).
- Running on more than one machine. Everything happens on this server.
- Replacing `docs/project_plan.md` wholesale. The scientific details there are still correct; this change modifies the execution-plan parts only.
- Building a general-purpose agent-orchestration framework. The rules here are scoped to this one project.
- Changing the refusal classifier, abliteration math, or any numeric hyperparameter from the original plan.

## Decisions

### Decision 1: Worktree-per-workstream topology
**Choice**: Each science workstream lives in its own git worktree + feature branch, all rooted under a sibling directory `../geometry-of-alignment-worktrees/`.

Branches and worktree paths:

| Branch | Worktree path | Agent scope |
|---|---|---|
| `agent/env-bootstrap` | `../gb-env/` | M0: verify envs, hooks, smoke tests. Can be throwaway after M1. |
| `agent/benchmark-eval` | `../gb-bench/` | Runs `src/benchmark/evaluate.py` + analysis on all models. GPU-light (llama.cpp). |
| `agent/mechanistic-analysis` | `../gb-mech/` | Runs activation extraction, layer analysis, visualization. **GPU-heavy**. |
| `agent/abliteration` | `../gb-ablit/` | Runs abliterate, ablation_study, selective_safety. **GPU-heavy**. |
| `agent/weight-diff` | `../gb-wdiff/` | CPU-only Qwen weight diff + SVD + MoE analysis. |
| `agent/writeup` | `../gb-paper/` | M5 only: paper + slides. Starts empty, pulls all other branches after M4 gate. |

**Rationale**: worktrees let multiple agents have independent working copies of the same repo pointing at different branches without cloning. They share `.git/`, so committing and pushing works normally. Mechanistic-analysis and abliteration are the most contentious workstreams because both want the GPU; putting them in different worktrees ensures that "one is running a sweep" doesn't mean "the other can't even edit a file".

**Alternative considered**: Single worktree, branch switching. Rejected — branch switching stomps uncommitted work and breaks multi-agent parallelism outright.

**Alternative considered**: Clone the repo N times. Rejected — wastes disk (models are big, though they live under `model/` which is gitignored), fragments git state, and makes pushing painful.

### Decision 2: Single-writer GPU lock via flock
**Choice**: Any agent about to instantiate a CUDA model (`transformers.AutoModel.from_pretrained` with `device_map="auto"` or `.cuda()`) MUST first acquire an exclusive flock on `~/.geometry-of-alignment/.gpu.lock`. Lock timeout is 6 hours; longer jobs must chunk themselves. A tiny helper `scripts/gpu_lock.sh` wraps `flock -x -w 60 ~/.geometry-of-alignment/.gpu.lock -c "<command>"`.

**Rationale**: Simple, file-based, kernel-enforced. Works across tmux panes, Claude Code agents, and manual sessions. No daemon, no registry, no partial ordering. If an agent dies the kernel releases the lock. 16 GB is too tight to run two Gemma 4 E4B sessions; a hard mutex is the only safe option.

**Alternative considered**: Time-sharing by MIG or CUDA MPS. Rejected — RTX 40-series cards don't support MIG, and MPS makes OOM crashes harder to debug.

**Alternative considered**: Advisory "first agent wins" convention. Rejected — a convention that depends on two LLM agents correctly polling a status file is exactly the kind of coordination that fails silently.

### Decision 3: Agent dispatch contract
**Choice**: When the operator dispatches an agent (manually via Claude Code or via the `Agent` tool), the dispatch prompt must include exactly these fields:

1. **Worktree path** — absolute path to the agent's working directory.
2. **Branch name** — `agent/<workstream>` — already checked out in the worktree.
3. **Tasks to execute** — a specific section ID from `tasks.md` (e.g. "execute section 4 (benchmark eval) until all sub-tasks are checked off").
4. **GPU policy** — one of `{gpu-none, gpu-lock-required}`. GPU-lock-required agents MUST wrap model-loading calls in `scripts/gpu_lock.sh`.
5. **Commit protocol** — commit after each checked-off task with `<type>: <what>` messages following the global git convention in `~/.claude/rules/common/git-workflow.md`. Push the branch after each commit so crashes don't lose work.
6. **Stop condition** — stop at the section boundary OR when `STATUS_FOR_HUMAN.md` needs a new section written. Do not jump into other sections.

**Rationale**: The dispatch contract is the only thing preventing agents from scope-creeping across workstreams and creating merge conflicts. The contract is enforced by prompt, not by tooling; it is documented in `tasks.md` so every dispatch can reference it.

### Decision 4: Human verification gate before writeup
**Choice**: After milestones M1–M3 are complete, agents produce `STATUS_FOR_HUMAN.md` at repo root on a dedicated `agent/writeup` branch, describing:

- Which experiments ran, with commit hashes and branch names.
- Refusal rates table (copy from `results/refusal_rates/`), any anomalies.
- Layer signal strength plot summary (did peak layers match the sliding/global hypothesis?).
- Ablation sweep summary (did alpha sweep behave monotonically? did random-direction control stay aligned?).
- Weight diff summary (was the Qwen diff low-rank? what fraction of params changed?).
- **"What the human needs to do"** checklist: eyeball `results/figures/*.png`, manually sanity-check N=10 responses from the abliterated model, approve proceeding to M5.
- **Explicit green-light sentence** that the writeup agent will grep for to know verification passed.

No writeup work happens until the operator has written the green-light sentence into `STATUS_FOR_HUMAN.md` or into a follow-up message. This is the one place in the workflow where a human *must* be in the loop.

**Rationale**: The paper makes factual claims about what the experiments showed. If the experiments are broken (e.g., refusal classifier has a bug and every response is mislabeled), the paper will confidently repeat the broken claims. Forcing a human to look at the numbers once before any prose is written is cheap insurance.

### Decision 5: Milestone schedule without wall-clock dates
**Choice**: Replace the original "Week 1 / Week 2 / ..." calendar with artifact-gated milestones:

- **M0 — Environment**: Python deps installed, GPU lock script in place, worktrees created, all agents can `python -c "import torch; torch.cuda.is_available()"` successfully. Smoke test: load Gemma 4 E2B in BF16 inside the mechanistic worktree, one prompt in, one token out.
- **M1 — Benchmark frozen**: `data/benchmark_prompts.json` passes schema validation, prompt count per category meets targets, committed on main. Everyone blocks on this before heavy work.
- **M2 — GPU experiments complete**: All of Section 4 (benchmark on GPU), Section 5 (mechanistic), Section 6 (abliteration + selective safety) tasks in `alignment-geometry-study/tasks.md` are checked off on their respective agent branches. Figures produced.
- **M3 — CPU weight diff complete**: Section 7 (weight diff + SVD + MoE) tasks checked off on `agent/weight-diff`.
- **M4 — Human verification gate**: `STATUS_FOR_HUMAN.md` written on `agent/writeup`, operator has reviewed and written the green-light sentence. No writeup before this.
- **M5 — Paper + slides**: Section 8 (writeup) completes. Final paper .tex / .md and slide deck pushed.

**Rationale**: The original dates assumed human-hours. Agents don't consume human-hours the same way; whether M2 takes 6 hours or 6 days depends on GPU sweep settings, not on calendar. Artifact gates are the only deadlines that matter.

**Alternative considered**: Keep calendar dates as "target by". Rejected — invites premature hand-offs and makes status reports noisier.

### Decision 6: Small scope for spec deltas
**Choice**: This change adds exactly one new capability (`autonomous-execution`) and modifies exactly one existing capability (`research-paper`). It does *not* touch `benchmark-evaluation`, `activation-analysis`, `abliteration-engine`, or `weight-diff-analysis`, because none of their *requirements* change — only who runs them.

**Rationale**: Minimizing spec delta surface area makes the change easier to archive and less likely to conflict with the still-open `alignment-geometry-study` change. The spec deltas tell the "what the system must do" story; the execution topology is a "how we run it" story that lives in `design.md` and `tasks.md`.

### Decision 7: Crash resilience — push after every task
**Choice**: Every agent pushes its feature branch after every completed sub-task in `tasks.md`. Commits are small and frequent. Agents never `git reset --hard` or `git push --force`. If a task fails, the agent opens a follow-up commit `fix: <what>` rather than amending.

**Rationale**: Claude Code sessions can be killed. We want zero work loss. Frequent pushes also make it possible for the operator to tail progress by watching the remote in another terminal.

## Risks / Trade-offs

- **[Two GPU-touching agents race before acquiring lock]** → flock is atomic; losing agent waits. If both wait longer than 6 hours, both fail cleanly and the operator sees `flock: timeout` in logs, not silent corruption.
- **[Agent scope-creeps across workstreams and creates merge conflicts]** → dispatch contract pins each agent to a single section of `tasks.md`. Worktrees prevent accidental cross-branch edits.
- **[Disk usage balloons with 6 worktrees]** → worktrees share object storage with the main clone. Only the working copy is duplicated. `model/` is gitignored, so the 15 GB of Gemma weights exist once on disk.
- **[An agent marks a task complete without actually running it]** → commits are frequent and small, so the operator (or a reviewer agent) can re-run smoke tests per commit. M4 gate requires figures to exist — if they don't, the gate fails.
- **[Writeup agent hallucinates experiment results]** → writeup agent is explicitly instructed to quote numbers only from files under `results/`, and to cite the commit hash + file path for every claim. M4 `STATUS_FOR_HUMAN.md` is the ground truth; the paper must be traceable to it.
- **[Human forgets to write the green-light sentence and the writeup agent hangs]** → not a risk: the writeup agent polls for the sentence at dispatch time and exits cleanly if absent, asking the operator to run it again after verification.
- **[Agent pushes a commit that breaks `main` for another agent]** → agents never touch `main`. `main` only advances through human-directed merges at M1 (benchmark freeze) and M4/M5 (final).
- **[Instructor authorization changes or is rescinded]** → recorded in `MEMORY.md`; operator must explicitly deauthorize before agents stop. Not a design concern.

## Migration Plan

1. Create the worktrees and branches listed in Decision 1. Do this once, manually, from the main clone.
2. Write `scripts/gpu_lock.sh` as a small flock wrapper. Add it to `main` and let every worktree inherit it via branch-off.
3. Add a pointer at the top of `docs/project_plan.md` redirecting execution-plan readers here.
4. Dispatch M0 agent in `agent/env-bootstrap` worktree. When it reports green, move on.
5. Dispatch M1 "benchmark freeze" on `main` (or on `agent/env-bootstrap` — it's a doc-level task, either works). Merge to `main` when done.
6. Dispatch M2 and M3 agents in parallel (one GPU, one CPU), each pinned to their worktree and tasks section.
7. Dispatch M4 status-writer on `agent/writeup`. Wait for operator sign-off.
8. Dispatch M5 paper/slides agent on `agent/writeup`.
9. Merge `agent/writeup` to `main` as the final deliverable branch.

## Open Questions

- Should the abliteration worktree use a conda env with a pinned bitsandbytes version separate from mechanistic-analysis? (Leaning no — same box, same env, same CUDA. Pin at M0.)
- Where does MMLU/GSM8K capability evaluation run — in the abliteration worktree or benchmark worktree? (Leaning abliteration, since it uses abliterated models and already has the GPU lock.)
- Does the M4 status doc need its own spec scenarios, or is the `autonomous-execution` requirement enough? (Current answer: one scenario under `autonomous-execution` is enough; don't multiply capabilities.)
