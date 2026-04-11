## 1. M0 — Environment Bootstrap

Goal: every worktree can load models, the GPU lock exists, all dependencies installed.
Agent scope: `agent/env-bootstrap` worktree only. GPU policy: gpu-none for steps 1.1–1.6; gpu-lock-required for 1.7.

- [ ] 1.1 Create `~/.geometry-of-alignment/` state directory and touch `.gpu.lock` there
- [ ] 1.2 Write `scripts/gpu_lock.sh` — wraps a command in `flock -x -w 60 ~/.geometry-of-alignment/.gpu.lock -c "<cmd>"`, with a kill-after-6-hours timeout option
- [ ] 1.3 Add a pointer at the top of `docs/project_plan.md` redirecting execution-plan readers to `openspec/changes/autonomous-agent-pivot/`; leave the scientific content of `project_plan.md` intact
- [ ] 1.4 Create the six worktrees listed in design.md Decision 1 under `../geometry-of-alignment-worktrees/` with `git worktree add ../gb-<name> -b agent/<branch>` for each. Do this from the main clone before any agents are dispatched.
- [ ] 1.5 Verify `python -c "import torch, transformers, bitsandbytes; print(torch.cuda.is_available())"` returns True in every worktree's venv (or the shared venv if one is used project-wide)
- [ ] 1.6 Verify `huggingface-cli` is logged in and both `model/gemma-4-E4B-it/` and `model/gemma-4-E2B-it/` exist and have `model.safetensors`
- [ ] 1.7 Smoke test (gpu-lock-required): in `../gb-mech/`, run `scripts/gpu_lock.sh python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; m=AutoModelForCausalLM.from_pretrained('./model/gemma-4-E2B-it', torch_dtype='bfloat16').cuda(); t=AutoTokenizer.from_pretrained('./model/gemma-4-E2B-it'); print(t.decode(m.generate(**t('hi', return_tensors=\"pt\").to('cuda'), max_new_tokens=5)[0]))"` — must print a short response and exit cleanly
- [ ] 1.8 Commit `scripts/gpu_lock.sh` and the `project_plan.md` pointer on `main` (this is a bootstrap exception — safe because it's docs + a harness script, no science)
- [ ] 1.9 Declare M0 complete by writing a single-line marker `M0=done` in `STATUS_FOR_HUMAN.md` on `agent/writeup` (create the file if needed)

## 2. M1 — Benchmark Freeze

Goal: `data/benchmark_prompts.json` is final and committed to `main`. Everyone else blocks on this.
Agent scope: `main` (short-lived bootstrap exception, doc-level only). GPU policy: gpu-none.

- [ ] 2.1 Validate that `data/benchmark_prompts.json` already exists, is valid JSON, and every prompt has fields `id, category, expected, prompt, variants`
- [ ] 2.2 Count prompts per category; verify targets from `alignment-geometry-study/tasks.md` sections 2.2–2.9 are met (50+ emergency_medical, 50+ wilderness_survival, 40+ each for home_safety, chemistry_safety, mental_health, gray_zone, should_refuse, safe_control)
- [ ] 2.3 If any category is under-filled, add prompts to meet the target (use the variant templates from `docs/project_plan.md`)
- [ ] 2.4 Commit the final `data/benchmark_prompts.json` on `main` with message `chore: freeze benchmark_prompts.json for M1`
- [ ] 2.5 Tag `main` with `m1-benchmark-frozen` so other agents can check it out exactly
- [ ] 2.6 Update `STATUS_FOR_HUMAN.md` — append `M1=done` and the commit hash

## 3. M2a — Benchmark Evaluation Pipeline

Goal: execute section 3 of `alignment-geometry-study/tasks.md` — run benchmark evaluation across all models.
Agent scope: `agent/benchmark-eval` worktree. GPU policy: gpu-none (llama.cpp CPU) unless a task specifically requires transformers loading.

- [ ] 3.1 In `../gb-bench/`, rebase onto the `m1-benchmark-frozen` tag
- [ ] 3.2 Execute tasks 3.1–3.3 from `alignment-geometry-study/tasks.md` (implement classify_refusal.py, evaluate.py, transformers backend). Code is already scaffolded in `src/benchmark/` — verify and fill gaps only.
- [ ] 3.3 Smoke test: run evaluate.py on 10 prompts against Gemma 4 E4B (GGUF via llama.cpp) before any full sweep. Commit a smoke-test log to `results/refusal_rates/.smoke/`.
- [ ] 3.4 Execute `alignment-geometry-study/tasks.md` tasks 3.4–3.10 (full evaluations on E4B, E2B, Qwen3.5-35B-A3B original and uncensored). Commit and push after each model.
- [ ] 3.5 Execute tasks 3.11–3.13 (phrasing sensitivity, context sensitivity, analyze_results.py figures). Produce `results/figures/refusal_heatmap.png`, `phrasing_sensitivity.png`.
- [ ] 3.6 Task 3.14 is deferred to M2c (after abliteration finishes). Leave unchecked in the source tasks.md.
- [ ] 3.7 Push branch and update `STATUS_FOR_HUMAN.md` with commit hash + a one-line summary of headline refusal rates

## 4. M2b — Mechanistic Analysis (GPU)

Goal: execute section 4 of `alignment-geometry-study/tasks.md` — activation extraction, layer analysis, visualization.
Agent scope: `agent/mechanistic-analysis` worktree. GPU policy: gpu-lock-required for all model-loading steps.

- [ ] 4.1 In `../gb-mech/`, rebase onto `m1-benchmark-frozen`
- [ ] 4.2 Execute `alignment-geometry-study/tasks.md` tasks 4.1–4.3 (ActivationCollector + smoke test on E2B BF16). Wrap model loading in `scripts/gpu_lock.sh`. Commit after smoke test.
- [ ] 4.3 Execute tasks 4.4–4.7 (full activation extraction for refuse + comply on Gemma 4 E4B 8-bit, compute refusal directions). Acquire GPU lock for the duration. If the run exceeds 5 hours, chunk it into smaller pushes.
- [ ] 4.4 Execute tasks 4.8–4.11 (signal strength per layer, PCA rank analysis). Produce `results/figures/signal_vs_layer.png`, `pca_variance_per_layer.png`.
- [ ] 4.5 Execute tasks 4.12–4.13 (UMAP/t-SNE visualization grid). Produce `results/figures/umap_layer_*.png`.
- [ ] 4.6 Execute task 4.14 (cross-precision validation on E2B BF16). Document cosine similarity with E4B directions.
- [ ] 4.7 Push branch and update `STATUS_FOR_HUMAN.md` with: peak layer indices, whether global layers dominated, rank-1 hypothesis result

## 5. M2c — Abliteration + Selective Safety (GPU)

Goal: execute section 5 of `alignment-geometry-study/tasks.md` — abliteration, ablation sweeps, selective safety.
Agent scope: `agent/abliteration` worktree. GPU policy: gpu-lock-required.

Dependency: M2b must be past task 4.3 (refusal directions exist) before 5.5 below starts.

- [ ] 5.1 In `../gb-ablit/`, rebase onto `m1-benchmark-frozen`
- [ ] 5.2 Verify `src/abliterate/abliterate.py`, `ablation_study.py`, `selective_safety.py` scaffolding is runnable. Fill gaps if present.
- [ ] 5.3 Execute `alignment-geometry-study/tasks.md` tasks 5.1–5.4 (implementation and E2B sanity test). Use gpu-lock for model loads.
- [ ] 5.4 Wait on `results/refusal_directions.pt` from the mechanistic worktree. Pull the latest `agent/mechanistic-analysis` branch into a read-only checkout (or wait for the operator to cherry-pick), then copy refusal_directions.pt into `../gb-ablit/results/activations/`.
- [ ] 5.5 Execute task 5.5–5.6 (full abliteration of E4B at alpha=1.0, quick-test refusal removal). Commit abliterated model safetensors under `models/gemma-4-e4b-abliterated/` (gitignored — record path in commit message only).
- [ ] 5.6 Execute tasks 5.7–5.11 (ablation sweeps: alpha, layer subset, prompt count, random direction control). Each sweep commits incrementally to `results/ablation_results/`.
- [ ] 5.7 Execute task 5.12 (capability preservation: MMLU + GSM8K subsets). Commit per-model capability scores.
- [ ] 5.8 Execute tasks 5.13–5.16 (category-specific directions, selective abliteration, results figures). Produce `results/figures/alpha_sweep.png`, `layer_subset_comparison.png`, `selective_safety_table.md`.
- [ ] 5.9 Hand-off to benchmark agent: notify via commit message that abliterated models are ready for M2c-followup task 3.14 (benchmark eval on abliterated models).
- [ ] 5.10 Push branch and update `STATUS_FOR_HUMAN.md` with: alpha curve shape, whether selective safety worked, capability delta vs original

## 6. M2c-followup — Benchmark on Abliterated Models

Goal: execute `alignment-geometry-study/tasks.md` task 3.14 now that abliterated models exist.
Agent scope: `agent/benchmark-eval` worktree (return to it after M2b/M2c done). GPU policy: gpu-none unless transformers backend is needed.

- [ ] 6.1 Merge-pull the abliterated model paths from `agent/abliteration` (via operator or cherry-pick)
- [ ] 6.2 Execute task 3.14 — run benchmark evaluate.py on the abliterated + selectively abliterated models
- [ ] 6.3 Regenerate `results/figures/refusal_heatmap.png` including the abliterated rows
- [ ] 6.4 Push branch and update `STATUS_FOR_HUMAN.md`

## 7. M3 — Weight Diff (CPU, parallel with M2)

Goal: execute section 6 of `alignment-geometry-study/tasks.md` — Qwen weight diff + SVD + MoE analysis.
Agent scope: `agent/weight-diff` worktree. GPU policy: gpu-none. This runs concurrently with M2 because it never touches the GPU.

- [ ] 7.1 In `../gb-wdiff/`, rebase onto `m1-benchmark-frozen`
- [ ] 7.2 Ensure the Qwen3.5-35B-A3B original and uncensored model files are downloaded under `model/` (gitignored). Log paths.
- [ ] 7.3 Execute `alignment-geometry-study/tasks.md` tasks 6.1–6.5 (load, compute diffs, report fraction changed)
- [ ] 7.4 Execute tasks 6.6–6.7 (per-layer Frobenius norm chart, singular value spectra). Produce `results/figures/weight_diff_per_layer.png`.
- [ ] 7.5 Execute tasks 6.8–6.11 (MoE expert analysis, expert heatmap, router modification report)
- [ ] 7.6 Execute task 6.12 (cross-reference with mechanistic refusal directions — qualitative, across models)
- [ ] 7.7 Execute task 6.13 (all final figures for Section 7)
- [ ] 7.8 Push branch and update `STATUS_FOR_HUMAN.md` with: fraction of params changed, low-rank yes/no, whether router was modified

## 8. M3b — Literature Survey (optional parallel)

Goal: execute section 7 of `alignment-geometry-study/tasks.md` — literature survey. This is text work; it can run on `agent/weight-diff` or `agent/writeup` at operator's discretion.
GPU policy: gpu-none.

- [ ] 8.1 Execute `alignment-geometry-study/tasks.md` tasks 7.1–7.5 (RLHF/DPO/CAI, representation engineering, abliteration, over-refusal, 15+ citations). Output to `paper/sections/02_background.md` and `paper/sections/03_related_work.md`.
- [ ] 8.2 Push and update `STATUS_FOR_HUMAN.md`

## 9. M4 — Human Verification Gate

Goal: produce a complete `STATUS_FOR_HUMAN.md` and wait for the operator's green light.
Agent scope: `agent/writeup` worktree. GPU policy: gpu-none.

- [ ] 9.1 Pull the latest commit hashes from all `agent/*` branches into a summary table at the top of `STATUS_FOR_HUMAN.md`
- [ ] 9.2 Write section (a) "Branch and commit status"
- [ ] 9.3 Write section (b) "Refusal rates table" — copy from `results/refusal_rates/` files. Cite the exact file path for each row.
- [ ] 9.4 Write section (c) "Mechanistic analysis summary" — peak layer indices, signal strength plot filename, rank-1 verdict
- [ ] 9.5 Write section (d) "Abliteration sweep summary" — alpha curve shape, layer subset comparison, selective safety verdict, capability delta
- [ ] 9.6 Write section (e) "Weight diff summary" — fraction params changed, low-rank verdict, MoE router verdict
- [ ] 9.7 Write section (f) "What the human needs to do" — a numbered checklist. Minimum items:
    - Open every `results/figures/*.png` and eyeball it for obvious breakage
    - Read 10 random responses from the abliterated model and confirm they are plausible
    - Verify `results/refusal_rates/gemma4_e4b_original.csv` row for `should_refuse` category has refusal rate > 80%
    - Verify `results/refusal_rates/gemma4_e4b_abliterated.csv` row for `should_refuse` category has refusal rate < 30% (i.e. abliteration worked)
    - Verify no credentials leaked into any committed file (grep for `HF_TOKEN`, `HUGGING_FACE`, `API_KEY`)
    - Decide whether any branch needs to be re-run or is good to merge
    - Write the green-light sentence into `STATUS_FOR_HUMAN.md` or into the next chat turn: **"Approved to proceed to M5 — writeup authorized."**
- [ ] 9.8 Write section (g) "Known anomalies or deviations from plan" — any surprises, broken assumptions, pivots
- [ ] 9.9 Commit and push `STATUS_FOR_HUMAN.md` on `agent/writeup`
- [ ] 9.10 STOP and wait for the operator. Do not start M5 without the green-light sentence.

## 10. M5 — Paper + Slides

Goal: execute section 8 of `alignment-geometry-study/tasks.md` — paper sections 1–9 and slide deck.
Agent scope: `agent/writeup` worktree. GPU policy: gpu-none.

**PRECONDITION**: `STATUS_FOR_HUMAN.md` on `agent/writeup` contains the sentence "Approved to proceed to M5 — writeup authorized." If this sentence is absent, the M5 agent MUST stop immediately.

- [ ] 10.1 Verify the green-light sentence exists; if not, stop and report M4 unresolved
- [ ] 10.2 Execute `alignment-geometry-study/tasks.md` task 8.1 (Section 1 Introduction) — use the hiking emergency scenario from `docs/project_plan.md`
- [ ] 10.3 Execute task 8.2 (Section 4 Over-Refusal Analysis) — quote only numbers that exist in `results/refusal_rates/`, cite file paths
- [ ] 10.4 Execute task 8.3 (Section 5 Mechanistic Analysis) — cite figures from `results/figures/`
- [ ] 10.5 Execute task 8.4 (Section 6 Abliteration & Selective Safety) — cite `results/ablation_results/` and `results/figures/`
- [ ] 10.6 Execute task 8.5 (Section 7 Weight Diff Analysis) — cite `results/weight_diffs/`
- [ ] 10.7 Execute task 8.6 (Section 8 Discussion & Course Connections) — over-parametrization, matrix perturbation, NTK
- [ ] 10.8 Execute task 8.7 (Section 9 Conclusion & Ethics)
- [ ] 10.9 Execute task 8.8 (integrate all sections, compile LaTeX or final markdown)
- [ ] 10.10 Execute task 8.9 (slide deck: hiking scenario opener, 3+ key figures, course-connections slide)
- [ ] 10.11 Produce a companion `paper/sources.md` mapping every numeric claim in the paper to its source file path and commit hash
- [ ] 10.12 Execute task 8.10 (practice run-through — for an agent this means re-read and self-critique for consistency/formatting errors)
- [ ] 10.13 Execute task 8.11 (final review) and write `READY_FOR_SUBMISSION.md` flagging the paper + slides as complete
- [ ] 10.14 Push final commits on `agent/writeup` and stop. The operator decides when to merge to `main`.

## 11. Cleanup / Hand-off

- [ ] 11.1 Operator (human) reviews `READY_FOR_SUBMISSION.md` and any final agent reports
- [ ] 11.2 Operator merges `agent/writeup` to `main` (or asks an agent to do it with explicit authorization)
- [ ] 11.3 Optionally remove worktrees with `git worktree remove ../gb-<name>` after merge. Branches stay on origin as provenance.
- [ ] 11.4 Archive this change (`autonomous-agent-pivot`) and `alignment-geometry-study` when done: `openspec archive --change autonomous-agent-pivot` and same for the other
