## ADDED Requirements

### Requirement: Worktree-per-workstream topology
The project SHALL execute each science workstream inside a dedicated git worktree on a dedicated feature branch. Agents SHALL operate only within their assigned worktree and SHALL NOT modify files in other worktrees or on `main` directly.

Required branches and worktrees:

| Branch | Worktree path | GPU policy |
|---|---|---|
| `agent/env-bootstrap` | `../gb-env/` | gpu-none |
| `agent/benchmark-eval` | `../gb-bench/` | gpu-none (llama.cpp on CPU) |
| `agent/mechanistic-analysis` | `../gb-mech/` | gpu-lock-required |
| `agent/abliteration` | `../gb-ablit/` | gpu-lock-required |
| `agent/weight-diff` | `../gb-wdiff/` | gpu-none (CPU-only) |
| `agent/writeup` | `../gb-paper/` | gpu-none |

#### Scenario: Agent dispatched to worktree
- **WHEN** an agent is dispatched to work on the mechanistic analysis workstream
- **THEN** it SHALL be launched with `cwd=../gb-mech/`, branch `agent/mechanistic-analysis` already checked out, and a prompt referencing only section 4 of `tasks.md`

#### Scenario: Agent never touches main
- **WHEN** any agent completes a sub-task
- **THEN** it SHALL commit to its own `agent/*` branch, never directly to `main`, and SHALL NOT perform destructive operations such as `git reset --hard`, `git push --force`, or `git checkout -- <file>`

### Requirement: GPU single-writer lock
Any code path that loads a model onto CUDA (including `transformers.AutoModel.from_pretrained(..., device_map="auto")`, explicit `.cuda()`, `.to("cuda")`, or `bitsandbytes` 8-bit loading) SHALL first acquire an exclusive file lock on `~/.geometry-of-alignment/.gpu.lock` with timeout 60 seconds and max hold 6 hours. The lock SHALL be released when the CUDA model is freed or the process exits.

#### Scenario: Two GPU agents run simultaneously
- **WHEN** the mechanistic agent is mid-run holding the GPU lock, and the abliteration agent starts an alpha sweep that needs CUDA
- **THEN** the abliteration agent SHALL block on flock until the mechanistic agent releases, then proceed, and SHALL NOT produce an OOM crash or share the card

#### Scenario: Lock expires on crash
- **WHEN** a GPU-holding agent process is killed or crashes
- **THEN** the kernel SHALL release the flock automatically so the next agent can proceed without manual intervention

### Requirement: Agent dispatch contract
Every agent dispatch prompt SHALL include: (1) absolute worktree path, (2) branch name, (3) a specific tasks.md section ID that bounds the agent's scope, (4) GPU policy tag, (5) commit-and-push protocol with commit-message format, (6) stop condition at section boundary, (7) model selection (`claude-sonnet-4-6` for mechanical/well-specified work, `claude-opus-4-7` for tasks demanding novel decisions, ambiguity resolution, or writing).

#### Scenario: Scope-bounded dispatch
- **WHEN** an agent is dispatched to execute "tasks.md section 5 (abliteration)"
- **THEN** it SHALL only modify files within `src/abliterate/`, `results/ablation_results/`, `results/figures/` for abliteration, and `models/` for its own abliterated model outputs, and SHALL NOT touch other sections' files

#### Scenario: Commit after every sub-task
- **WHEN** an agent checks off a sub-task in its scoped tasks.md section
- **THEN** it SHALL create a commit with a message following `<type>: <description>` format and SHALL push the commit to the remote `agent/*` branch before starting the next sub-task

#### Scenario: Model selected per task profile
- **WHEN** the dispatcher selects a model for a section
- **THEN** Sonnet 4.6 (`claude-sonnet-4-6`) SHALL be the default for mechanical work — running existing scripts, parameter sweeps, file/path shuffling, rebase/push housekeeping, well-specified evaluations (e.g., M2a 3.5–3.9, M2c 5.7 sweep, M3 7.7 weight-diff runs); Opus 4.7 (`claude-opus-4-7`) SHALL be the default where the task demands judgment calls or prose — interpreting M2b mechanistic results, writing the Gemma-quirk discussion in M2c 5.6, designing the M3 7.10 cross-reference, all of M5 (paper + slides). Per-dispatch overrides SHALL be recorded with a one-line rationale in the dispatch prompt.

### Requirement: Artifact-gated milestones
The project SHALL progress through milestones M0 through M5 without wall-clock deadlines. Each milestone SHALL be declared complete only when the listed artifacts exist on disk and the listed tasks are checked off.

- **M0 Environment**: Python env installed in every worktree, `scripts/gpu_lock.sh` present, CUDA available, Gemma 4 E2B BF16 loads successfully in a smoke test.
- **M1 Benchmark frozen**: `data/benchmark_prompts.json` passes schema validation, committed to `main`, every category meets the prompt count target from the original plan.
- **M2 GPU + benchmark experiments complete**: All tasks under sections 3 (M2a benchmark eval), 4 (M2b mechanistic), 5 (M2c abliteration), and 6 (M2c-followup benchmark on abliterated models) of `tasks.md` are checked off on their respective `agent/*` branches, with figures in `results/figures/`.
- **M3 Comparative weight diff complete**: All tasks under section 7 of `tasks.md` are checked off on `agent/weight-diff`, with `results/weight_diffs/` populated for at least the OBLITERATUS variant. **No MoE/router/expert artifacts are produced** — that workstream is removed.
- **M4 Human verification gate**: `STATUS_FOR_HUMAN.md` written on `agent/writeup`, operator has written the green-light sentence.
- **M5 Paper + slides**: Paper draft and slides produced on `agent/writeup`.

#### Scenario: Milestone M2 declared complete
- **WHEN** the operator asks whether M2 is done
- **THEN** an agent SHALL verify by running a checklist: every sub-task in sections 3, 4, 5, 6 of `tasks.md` is `[x]`, every expected file in `results/` exists and is non-empty, and every required figure PNG exists in `results/figures/`, before answering yes

### Requirement: Human verification gate before writeup
Before any paper or slide work begins, the project SHALL produce a `STATUS_FOR_HUMAN.md` document at repo root on the `agent/writeup` branch that summarizes every completed experiment, links to every generated figure, lists any anomalies, and contains an explicit "what the human needs to do" checklist. No writeup (paper prose, slides) SHALL be written until the operator has added a green-light sentence to `STATUS_FOR_HUMAN.md` or to a chat response.

#### Scenario: Writeup agent blocked without green light
- **WHEN** the writeup agent is dispatched but no green-light sentence is present in `STATUS_FOR_HUMAN.md` or in the dispatch message
- **THEN** the writeup agent SHALL stop immediately, report that M4 has not passed, and SHALL NOT write any paper content

#### Scenario: Writeup agent proceeds after green light
- **WHEN** the writeup agent is dispatched and `STATUS_FOR_HUMAN.md` contains the sentence "Approved to proceed to M5 — writeup authorized." added by the operator
- **THEN** the writeup agent SHALL proceed to write Sections 1–9 of the paper and the slide deck, quoting only numbers that appear in files under `results/`

### Requirement: STATUS_FOR_HUMAN handoff artifact
`STATUS_FOR_HUMAN.md` SHALL contain at minimum: (a) list of agent branches with commit hashes and last-commit timestamps, (b) a refusal rates table copied from `results/refusal_rates/`, (c) the layer signal strength plot filename with one-line interpretation, (d) alpha sweep summary (did it behave monotonically? was the random-direction control flat?), (e) **comparative weight diff summary** (per variant: low-rank yes/no, fraction of params changed, top-1 cosine vs M2b refusal directions, shared-tensor de-dup count), (f) a numbered checklist of actions the human must take before M5 begins.

#### Scenario: STATUS_FOR_HUMAN fields present
- **WHEN** the M4 status-writer agent finishes
- **THEN** `STATUS_FOR_HUMAN.md` SHALL contain all six sections (a)–(f) and SHALL have at least one actionable item in the human checklist, or the agent SHALL report the gate as failed

### Requirement: Frequent push, no rewriting history
Agents SHALL push their feature branches after every completed sub-task. Agents SHALL NOT use `git commit --amend`, `git rebase -i`, or `git push --force` on any `agent/*` branch. If a commit is wrong, the fix SHALL be a new `fix:` commit on top.

#### Scenario: Agent crashes mid-sweep
- **WHEN** an agent crashes halfway through a 10-point alpha sweep after pushing commits for points 1–4
- **THEN** the next dispatched agent SHALL be able to resume from point 5 by reading the last commit on the branch, without any lost work

### Requirement: Deauthorization kill switch
If the auto-memory record `project_authorization.md` is removed or the operator explicitly says "stop autonomous work", every running agent SHALL halt at its next sub-task boundary and refuse to dispatch new agents until authorization is restored.

#### Scenario: Authorization rescinded
- **WHEN** the operator writes "pause autonomous work" in a dispatch session
- **THEN** the receiving agent SHALL finish its current commit, push it, and stop, without starting the next sub-task
