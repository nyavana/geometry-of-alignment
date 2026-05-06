# Tasks — Gemma-only execution plan

This is the single source of truth for milestones M0–M5 of the project. M0 and M1 are complete (history retained for reference). M2–M5 are the live workstreams.

**Dispatch contract** (carried forward from `autonomous-agent-pivot`): every agent dispatch SHALL include (1) absolute worktree path, (2) branch name, (3) a specific section ID from this file, (4) GPU policy, (5) commit-and-push protocol, (6) stop condition at section boundary.

**Environment activation** (every dispatched agent's first commands): `source /home/nyavana/columbia/6699/shared/env.sh && source .venv/bin/activate`. Without this, `python` is not on PATH.

**Path conventions:** Artifact paths written below as `results/...` resolve to `$RESULTS_DIR/...` (i.e., `/home/nyavana/columbia/6699/shared/results/agent/<branch>/...`), which is per-branch and outside any worktree's git repo. Cross-branch reads (e.g., M2c reading M2b's output) use the absolute form `/home/nyavana/columbia/6699/shared/results/agent/<source-branch>/...`. The `.gitignore` allowlist also permits a redundant in-repo copy of small handoff artifacts (refusal directions, summary JSONs/CSVs, headline figures) for git-based traceability — but the source of truth remains `$RESULTS_DIR`.

**Worktrees** (created in M0, names preserved):

| Branch | Worktree path | GPU policy |
|---|---|---|
| `agent/env-bootstrap` | `../gb-env/` | gpu-none |
| `agent/benchmark-eval` | `../gb-bench/` | gpu-none (llama-server CPU mode) or gpu-lock-required (llama-server -ngl) |
| `agent/mechanistic-analysis` | `../gb-mech/` | gpu-lock-required |
| `agent/abliteration` | `../gb-ablit/` | gpu-lock-required |
| `agent/weight-diff` | `../gb-wdiff/` | gpu-none (CPU-only) |
| `agent/writeup` | `../gb-paper/` | gpu-none |

---

## 1. M0 — Environment Bootstrap (COMPLETE)

Reference only — all sub-tasks done before this change was written.

- [x] 1.1 Create `~/.geometry-of-alignment/` state directory and touch `.gpu.lock` there
- [x] 1.2 Write `scripts/gpu_lock.sh` *(commit 13a711b)*
- [x] 1.3 Add pointer at top of `docs/project_plan.md` *(commit 13a711b)*
- [x] 1.4 Create the six worktrees under `../gb-<name>/` *(prior session)*
- [x] 1.5 Verify `python -c "import torch, transformers, bitsandbytes; print(torch.cuda.is_available())"` returns True in every worktree's venv *(verified — torch 2.11.0+cu130, transformers 5.5.3, bitsandbytes 0.49.2)*
- [x] 1.6 Verify both `model/gemma-4-E4B-it/` and `model/gemma-4-E2B-it/` exist with `model.safetensors` *(E4B=15G, E2B=9.6G; symlinked into each worktree)*
- [x] 1.7 GPU smoke test in `../gb-mech/` *(passed: bf16 on cuda:0, 5 tokens, lock acquired and released)*
- [x] 1.8 Commit `scripts/gpu_lock.sh` and the project_plan.md pointer on `main` *(commit 13a711b)*
- [x] 1.9 Declare M0 complete via `STATUS_FOR_HUMAN.md` marker *(commit 1aca11d on agent/writeup)*

## 2. M1 — Benchmark Freeze (COMPLETE)

Reference only.

- [x] 2.1 Validate `data/benchmark_prompts.json` schema
- [x] 2.2 Count prompts per category (initial: em=3, ws=2, sr=2, sc=2, others=0)
- [x] 2.3 Author 331 new prompts to reach 50/50/40/40/40/40/40/40 = 340 total + 640 variants. `scripts/build_benchmark.py` is the single source of truth
- [x] 2.4 Commit final `data/benchmark_prompts.json` *(commit cc8b7ec)*
- [x] 2.5 Tag `main` with `m1-benchmark-frozen` *(local tag; not pushed because origin remote is not yet configured)*
- [x] 2.6 Update `STATUS_FOR_HUMAN.md` with `M1=done` *(commit 40b0fe5 on agent/writeup)*

---

## 3. M2a — Benchmark Evaluation Pipeline

Goal: run benchmark evaluation across the new model lineup.
Agent scope: `agent/benchmark-eval` worktree (`../gb-bench/`). GPU policy: gpu-none if `llama-server` is launched without `-ngl` (CPU only) and for the OpenAI-compatible HTTP client; gpu-lock-required when `llama-server` is launched with `-ngl` to offload layers, and for the transformers backend.

**Model lineup (replacing the predecessor's Qwen rows):**
- `google/gemma-4-E4B-it` (base, GGUF + transformers)
- `google/gemma-4-E2B-it` (validation, BF16)
- `OBLITERATUS/gemma-4-E4B-it-OBLITERATED` (GGUF Q8_0 + bf16 safetensors)
- `TrevorJS/gemma-4-E4B-it-uncensored` (bf16 safetensors, transformers backend)
- `HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive` (GGUF Q8_K_P)
- (eventually) the project's own M2c abliterated E4B — added in section 6 below

- [ ] 3.1 In `../gb-bench/`, rebase onto `origin/main` (`m1-benchmark-frozen` is local-only and predates the llama-server backend swap; `origin/main` carries that swap, the prompt additions, and the path-convention preamble above)
- [ ] 3.2 Verify `src/benchmark/{evaluate.py,classify_refusal.py,analyze_results.py}` are runnable; fill any scaffolding gaps. `evaluate.py` MUST support both `--backend llamacpp` (HTTP client to `llama-server`, with `--server-url`) and `--backend transformers` (per the `benchmark-evaluation` spec).
- [x] 3.2.1 Install upstream llama.cpp's `llama-server` binary. *(Done from `main`: built from source against `apt install nvidia-cuda-toolkit` (CUDA 12.0) with `-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89`, installed to `/home/nyavana/columbia/6699/shared/llama.cpp-cuda/`. PATH + LD_LIBRARY_PATH are wired through `shared/env.sh`, so every worktree picks it up on `source shared/env.sh`. `llama-server --list-devices` confirms `CUDA0: NVIDIA GeForce RTX 4070 Ti SUPER (16375 MiB)`.)*
- [ ] 3.3 Smoke test: launch `llama-server -m /home/nyavana/columbia/6699/shared/model/gemma-4-E4B-it.Q8_0.gguf -ngl 99 --host 127.0.0.1 --port 8088` (8088 because Windows-side WSL2 binds 8080), then run `evaluate.py --backend llamacpp --server-url http://127.0.0.1:8088` on 10 prompts against base Gemma 4 E4B GGUF. Verified end-to-end working from `main` (43/43 layers offloaded to GPU, OpenAI chat-completions endpoint returns expected output). Commit smoke log to `results/refusal_rates/.smoke/`.
- [ ] 3.4 Manually label 50 sampled responses; iterate `classify_refusal.py` patterns until ≥90% agreement.
- [ ] 3.5 Full evaluation: base `google/gemma-4-E4B-it` (GGUF). `--output $RESULTS_DIR/refusal_rates/gemma4_e4b_base/` (writes `evaluation_results.{json,csv}`).
- [ ] 3.6 Full evaluation: `google/gemma-4-E2B-it` (BF16, validation). `--output $RESULTS_DIR/refusal_rates/gemma4_e2b_base/`.
- [ ] 3.7 Full evaluation: `OBLITERATUS/gemma-4-E4B-it-OBLITERATED` (GGUF Q8_0). `--output $RESULTS_DIR/refusal_rates/gemma4_e4b_obliteratus/`.
- [ ] 3.8 Full evaluation: `TrevorJS/gemma-4-E4B-it-uncensored` (bf16 via transformers, gpu-lock-required). `--output $RESULTS_DIR/refusal_rates/gemma4_e4b_trevorjs/`.
- [ ] 3.9 Full evaluation: `HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive` (GGUF Q8_K_P). `--output $RESULTS_DIR/refusal_rates/gemma4_e4b_hauhau/`.
- [ ] 3.10 Phrasing sensitivity analysis: re-run on prompt variants for the base + at least one published variant.
- [ ] 3.11 Context sensitivity test: prepend "I am an emergency first responder" to `emergency_medical` prompts, compare refusal rates. Run on base + OBLITERATUS.
- [ ] 3.12 `analyze_results.py`: produce `results/figures/refusal_heatmap.png` (rows: 5 models above; columns: 8 categories) and `results/figures/phrasing_sensitivity.png`.
- [ ] 3.13 Push branch with a final commit whose message includes: per-model headline refusal rates, over-refusal rate on `emergency_medical`, and any phrasing- or context-sensitivity surprises. (M4 9.x reads these from `git log` across `agent/*` branches and assembles `STATUS_FOR_HUMAN.md`.)

*Note: evaluation of the project's own M2c-abliterated model is in section 6 below, after M2c lands.*

## 4. M2b — Mechanistic Analysis (GPU)

Goal: layer-by-layer activation extraction, refusal direction computation, layer-rank analysis, visualization. **Identical to the predecessor's section 4** — no changes from the swap.

Agent scope: `agent/mechanistic-analysis` worktree (`../gb-mech/`). GPU policy: gpu-lock-required for all model loads.

- [x] 4.1 In `../gb-mech/`, rebase onto `origin/main` (see preamble — `m1-benchmark-frozen` is stale).
- [ ] 4.2 Verify `src/mechanistic/{extract_activations.py,layer_analysis.py,visualize.py}` and the `ActivationCollector` class. Smoke test on E2B BF16 with 10 prompts inside `scripts/gpu_lock.sh`.
- [ ] 4.3 Full activation extraction on Gemma 4 E4B 8-bit: refuse-class (`should_refuse` + the five over-refuse categories `emergency_medical`, `wilderness_survival`, `home_safety`, `chemistry_safety`, `mental_health`) → `results/activations/refuse_activations.pt`; comply-class (`safe_control`) → `results/activations/comply_activations.pt`. Also write `results/activations/prompt_metadata.json` with the per-row mapping (`prompt_id`, `category`, `expected`) so M2c 5.9 can slice category-specific subsets without re-extracting on the GPU.
- [ ] 4.4 Compute refusal directions per layer via mean-diff → `$RESULTS_DIR/activations/refusal_directions.pt` (resolves to `/home/nyavana/columbia/6699/shared/results/agent/mechanistic-analysis/activations/refusal_directions.pt`). Also commit a redundant copy into `results/activations/refusal_directions.pt` (the `.gitignore` allowlist permits this) and push the branch. **CHECKPOINT — push immediately on completion. This is the M2b artifact that unblocks M2c task 5.4 and M3 task 7.10. Do not bundle 4.5–4.9 into the same commit.**
- [ ] 4.5 Signal-strength + sliding/global comparison → `results/figures/signal_vs_layer.png`.
- [ ] 4.6 PCA rank analysis per layer → `results/figures/pca_variance_per_layer.png`.
- [ ] 4.7 UMAP/t-SNE multi-layer grid → `results/figures/umap_layer_*.png`.
- [ ] 4.8 Cross-precision validation: refusal direction on E2B BF16, cosine similarity vs E4B 8-bit. Document.
- [ ] 4.9 Push branch with a final commit whose message includes: peak layer indices, sliding/global verdict, rank-1 hypothesis result. (M4 9.x reads these from `git log` and aggregates into `STATUS_FOR_HUMAN.md`.)

## 5. M2c — Abliteration + Selective Safety (GPU)

Goal: this project's own abliteration of Gemma 4 E4B + ablation sweeps + selective safety. **Identical to the predecessor's section 5**, with one addition: per recent literature, standard rank-1 abliteration may underperform on Gemma 4 due to RMSNorm + shared K/V — document the result either way.

Agent scope: `agent/abliteration` worktree (`../gb-ablit/`). GPU policy: gpu-lock-required.

Dependency: M2b task 4.4 (refusal directions exist) must be complete before 5.4 below starts.

- [ ] 5.1 In `../gb-ablit/`, rebase onto `origin/main` (see preamble — `m1-benchmark-frozen` is stale).
- [ ] 5.2 Verify `src/abliterate/{abliterate.py,ablation_study.py,selective_safety.py}` are runnable.
- [ ] 5.3 Sanity test on E2B BF16: extract directions, abliterate, verify refusal removal on 5 test prompts.
- [ ] 5.4 Read `refusal_directions.pt` directly from M2b's `$RESULTS_DIR`: `cp /home/nyavana/columbia/6699/shared/results/agent/mechanistic-analysis/activations/refusal_directions.pt $RESULTS_DIR/activations/refusal_directions.pt`. As a fallback if M2b's `$RESULTS_DIR` is empty (e.g., agent ran in a different machine), pull the in-repo redundant copy: `git fetch origin agent/mechanistic-analysis && git show origin/agent/mechanistic-analysis:results/activations/refusal_directions.pt > $RESULTS_DIR/activations/refusal_directions.pt`.
- [ ] 5.5 Full abliteration of E4B at alpha=1.0, all 42 layers. Save to `models/gemma-4-e4b-abliterated/` (gitignored — record path in commit message).
- [ ] 5.6 Quick-test: 20 benchmark prompts, confirm refusal removal works. **If refusal removal is incomplete, this is a paper finding** (Gemma 4 RMSNorm/shared-K/V resistance) — document and continue with sweeps.
- [ ] 5.7 **Sweep ensemble (single model load).** Run `python -m src.abliterate.ablation_study --model google/gemma-4-E4B-it --activations results/activations/ --benchmark data/benchmark_prompts.json --use-8bit --output results/ablation_results/`. The script loads the model + tokenizer once, snapshots the abliteration-target weights (`o_proj`, `down_proj` for all 42 layers), and runs alpha sweep + layer-subset sweep + random-direction control in sequence — restoring from the snapshot before each iteration to avoid disk reloads. Outputs: `alpha_sweep.json`, `layer_subset_sweep.json`, `random_direction_control.json` (and `sweep_results.json` aggregate). The prompt-count sweep `[10, 25, 50, 100, 200]` produces `prompt_count_sweep.json` — currently a follow-up since it requires recomputing directions per N rather than reusing the snapshot.
- [ ] 5.8 Capability preservation: MMLU + GSM8K subsets on original vs abliterated. → `results/ablation_results/capability_preservation.json`.
- [ ] 5.9 Category-specific refusal directions: load `results/activations/refuse_activations.pt` + `prompt_metadata.json` (from M2b 4.3), slice rows by `category` (`emergency_medical`, `wilderness_survival`, `should_refuse`), and compute each category's direction against the `safe_control` baseline. NO re-extraction on the GPU. Pairwise cosine similarity at each layer.
- [ ] 5.10 Selective abliteration: remove medical refusal direction only; eval over-refusal on medical (target: <10%) + refusal on should_refuse (target: >80%).
- [ ] 5.11 Figures: `results/figures/{alpha_sweep.png,layer_subset_comparison.png,selective_safety_table.md}`.
- [ ] 5.12 Hand-off: notify in commit message that abliterated models are ready for section 6 below.
- [ ] 5.13 Push branch with a final commit whose message includes: alpha curve shape, selective safety verdict, capability delta, and (if applicable) the Gemma 4 architectural-quirk failure note. (M4 9.x reads these from `git log` and aggregates into `STATUS_FOR_HUMAN.md`.)

## 6. M2c-followup — Benchmark on the project's own abliterated model

Goal: complete the heatmap by adding our own abliterated model.
Agent scope: `agent/benchmark-eval` worktree. GPU policy: gpu-lock-required for transformers backend.

- [ ] 6.1 Pull abliterated model paths from `agent/abliteration` (cherry-pick or operator-mediated).
- [ ] 6.2 Run `evaluate.py --backend transformers --use-8bit` against `models/gemma-4-e4b-abliterated/`. `--output $RESULTS_DIR/refusal_rates/gemma4_e4b_self_abliterated/`.
- [ ] 6.3 If selective abliteration produced a model in 5.10, evaluate that too. → `$RESULTS_DIR/refusal_rates/gemma4_e4b_self_selective/`.
- [ ] 6.4 Regenerate `results/figures/refusal_heatmap.png` including the new rows (now ~6–7 rows: base, E2B, OBLITERATUS, TrevorJS, HauhauCS, self-abliterated, self-selective).
- [ ] 6.5 Push branch with a final commit whose message includes: refusal rate on `should_refuse` for the self-abliterated model and (if produced) the self-selectively-abliterated model. (M4 9.x reads these from `git log` and aggregates into `STATUS_FOR_HUMAN.md`.)

## 7. M3 — Comparative Weight Diff (replaces Qwen MoE)

Goal: weight-diff between base Gemma 4 E4B-it and each published bf16 variant. SVD rank analysis. Cross-method comparison. Cross-reference with M2b refusal directions.

Agent scope: `agent/weight-diff` worktree (`../gb-wdiff/`). GPU policy: gpu-none. Runs concurrently with M2 (no GPU contention).

**Targets:**
- Primary: `OBLITERATUS/gemma-4-E4B-it-OBLITERATED` (bf16 safetensors)
- Secondary: `TrevorJS/gemma-4-E4B-it-uncensored` (bf16 safetensors source repo)
- Behavioral-only: `HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive` (GGUF — no weight diff possible; benchmark-eval coverage in section 3 above is sufficient)

**Dependencies:** Tasks 7.1–7.9 are independent of M2b and can run as soon as the worktree is ready. Task 7.10 (cross-reference with refusal directions) BLOCKS until `agent/mechanistic-analysis` has produced `refusal_directions.pt` (M2b task 4.4) — read it from `/home/nyavana/columbia/6699/shared/results/agent/mechanistic-analysis/activations/refusal_directions.pt`, or as fallback from `origin/agent/mechanistic-analysis:results/activations/refusal_directions.pt`.

- [ ] 7.1 In `../gb-wdiff/`, rebase onto `origin/main` (see preamble — `m1-benchmark-frozen` is stale).
- [ ] 7.2 **Pre-flight: disk + license check.**
  - Run `df -h /home/nyavana/columbia/6699/shared/` — confirm ≥40 GB free.
  - Read each variant's HuggingFace model card to verify license inheritance (OBLITERATUS card states Apache 2.0 from base; verify TrevorJS).
  - If insufficient disk OR a license blocker, stop and surface to operator.
- [ ] 7.3 Download `OBLITERATUS/gemma-4-E4B-it-OBLITERATED` bf16 safetensors via `huggingface-cli download` to `model/OBLITERATUS-gemma-4-E4B-it-OBLITERATED/`.
- [ ] 7.4 Download `TrevorJS/gemma-4-E4B-it-uncensored` bf16 safetensors to `model/TrevorJS-gemma-4-E4B-it-uncensored/`.
- [ ] 7.5 **Pre-flight: shape/key compatibility.** For each variant, load the state-dict header and assert keys match base; assert shapes match. If TrevorJS fails, log to `results/weight_diffs/.compat_log.md` and proceed with OBLITERATUS only (per design D2 fallback). If OBLITERATUS fails, stop and surface to operator.
- [ ] 7.6 Smoke-test `src/weight_diff/compute_diff.py` and `svd_analysis.py`: run against (base × OBLITERATUS) for one layer only. Confirm scripts produce JSON output and don't error.
- [ ] 7.7 **Full weight diff per variant.** For each variant in `[OBLITERATUS, TrevorJS]` that passed pre-flight 7.5, run `python -m src.weight_diff.compute_diff --original model/gemma-4-E4B-it/ --modified model/<variant>/ --output results/weight_diffs/<variant_slug>/` — produces per-parameter Frobenius/relative-change/max-abs-change JSON. Both runs are CPU-only and use ~34 GB RAM each — the agent MAY launch them in parallel (well within the 100 GB budget).
- [ ] 7.8 **SVD analysis per variant.** For each variant whose diff exists, run `python -m src.weight_diff.svd_analysis --results results/weight_diffs/<variant_slug>/weight_diff_results.json`. Produces effective rank at 95/99% and top-5 singular vectors per significantly-modified weight (saved as `.pt`). Parallelizable across variants.
- [ ] 7.9 **Cross-method comparison:** new analysis script (or extend `svd_analysis.py`):
  - Per-layer Frobenius bar chart with both methods overlaid → `results/figures/weight_diff_per_layer_overlay.png`.
  - For each significantly-modified parameter that exists in both: cosine similarity between top-1 left singular vectors (OBLITERATUS vs TrevorJS) → table `results/weight_diffs/cross_method_cosine_table.csv` and figure `results/figures/cross_method_singular_vectors.png`.
- [ ] 7.10 **Cross-reference with M2b refusal directions — BLOCKS until M2b 4.4 is pushed** (the *quantitative* version of original task 6.12):
  - For each layer where M2b computed a refusal direction AND M3 computed a top-1 left singular vector for the residual-stream-writing weights (`o_proj`, `down_proj`): compute cosine similarity.
  - → table `results/weight_diffs/refusal_direction_vs_singular_vector.csv` and figure `results/figures/refusal_direction_vs_singular_vector.png`.
- [ ] 7.11 **Architectural-quirk handling:** in the per-layer Frobenius chart and the cross-method tables, identify and de-duplicate the shared K/V tensors (per OBLITERATUS card: layers 24–41 reference layer 24's `k_proj`/`v_proj`). Each unique tensor counted once. Document in `results/weight_diffs/.shared_tensor_handling.md`.
- [ ] 7.12 Singular value spectrum plots for the most-modified weight matrices in each method. → `results/figures/singular_value_spectra_per_method.png`.
- [ ] 7.13 Component-type summary (attention vs MLP vs embedding vs norm) per method. → `results/weight_diffs/component_type_breakdown.csv`. (No MoE/expert/router rows — Gemma is dense.)
- [ ] 7.14 Push branch with a final commit whose message includes: low-rank verdict per method (rank-1? rank-3?), cosine similarity range vs M2b directions, shared-tensor de-dup count. (M4 9.x reads these from `git log` and aggregates into `STATUS_FOR_HUMAN.md`.)

## 8. M3b — Literature Survey (optional parallel)

Same as predecessor — drafted on `agent/weight-diff` or `agent/writeup`. GPU policy: gpu-none.

- [ ] 8.1 RLHF (Christiano 2017, Ouyang 2022), DPO (Rafailov 2023), Constitutional AI (Bai 2022).
- [ ] 8.2 Representation engineering (Zou 2023), linear representation hypothesis.
- [ ] 8.3 Abliteration: Arditi 2024, Heretic (p-e-w 2025), OBLITERATUS (elder-plinius 2025), grimjim's norm-preserving biprojection.
- [ ] 8.4 Over-refusal: Rottger 2024 (XSTest), Cui 2024.
- [ ] 8.5 Gemma 4 architectural quirks: source the "doesn't work on Gemma 4" findings (Heretic GitHub issues, OBLITERATUS card).
- [ ] 8.6 Output to `paper/sections/02_background.md` and `paper/sections/03_related_work.md`. Min 15 citations.

## 9. M4 — Human Verification Gate

Goal: produce `STATUS_FOR_HUMAN.md` and wait for operator's green-light sentence.
Agent scope: `agent/writeup`. GPU policy: gpu-none.

- [ ] 9.1 For each branch in {`agent/benchmark-eval`, `agent/mechanistic-analysis`, `agent/abliteration`, `agent/weight-diff`}, run `git log -1 origin/<branch>` and extract the headline numbers from the final commit message (the patterns documented in tasks 3.13, 4.9, 5.13, 6.5, 7.14). Aggregate into the section (a)–(e) entries below.
- [ ] 9.2 Section (a) "Branch and commit status."
- [ ] 9.3 Section (b) "Refusal rates table" — copy from `results/refusal_rates/`. Cite each row's source CSV.
- [ ] 9.4 Section (c) "Mechanistic analysis summary" — peak layer indices, signal strength plot, rank-1 verdict.
- [ ] 9.5 Section (d) "Abliteration sweep summary" — alpha curve shape, layer subset comparison, selective safety verdict, capability delta. **Include the Gemma 4 quirk note if 5.6 reported partial failure.**
- [ ] 9.6 Section (e) "Comparative weight diff summary" — for each method: low-rank verdict, fraction of params changed, top-1 cosine vs M2b refusal directions, shared-tensor de-dup count.
- [ ] 9.7 Section (f) "What the human needs to do":
  - Open every PNG under `/home/nyavana/columbia/6699/shared/results/agent/*/figures/` and eyeball for breakage.
  - Read 10 random responses from each abliterated model (self + OBLITERATUS + TrevorJS + HauhauCS) and confirm plausibility.
  - Verify `/home/nyavana/columbia/6699/shared/results/agent/benchmark-eval/refusal_rates/gemma4_e4b_base/evaluation_results.csv` shows `should_refuse` refusal rate >80%.
  - Verify `/home/nyavana/columbia/6699/shared/results/agent/benchmark-eval/refusal_rates/gemma4_e4b_self_abliterated/evaluation_results.csv` shows `should_refuse` refusal rate <30%.
  - Verify `/home/nyavana/columbia/6699/shared/results/agent/weight-diff/weight_diffs/cross_method_cosine_table.csv` exists and contains numeric values.
  - Verify `/home/nyavana/columbia/6699/shared/results/agent/weight-diff/figures/refusal_direction_vs_singular_vector.png` exists.
  - Grep for credentials leaks (`HF_TOKEN`, `HUGGING_FACE`, `API_KEY`).
  - Decide which branches to merge.
  - Write the green-light sentence: **"Approved to proceed to M5 — writeup authorized."**
- [ ] 9.8 Section (g) "Known anomalies or deviations from plan."
- [ ] 9.9 Commit and push `STATUS_FOR_HUMAN.md` on `agent/writeup`.
- [ ] 9.10 STOP. Do not start M5 without the green-light sentence.

## 10. M5 — Paper + Slides

Goal: 9-section paper + slide deck.
Agent scope: `agent/writeup`. GPU policy: gpu-none.

**PRECONDITION:** `STATUS_FOR_HUMAN.md` contains "Approved to proceed to M5 — writeup authorized."

- [ ] 10.1 Verify green-light sentence; if absent, stop and report M4 unresolved.
- [ ] 10.2 Section 1 (Introduction) — hiking emergency scenario.
- [ ] 10.3 Section 4 (Over-Refusal Analysis) — quote only numbers from `results/refusal_rates/`; cite CSV paths.
- [ ] 10.4 Section 5 (Mechanistic Analysis) — cite figures from `results/figures/`.
- [ ] 10.5 Section 6 (Abliteration & Selective Safety) — cite `results/ablation_results/` and `results/figures/`. **Include the Gemma 4 architectural-quirk discussion if 5.6 surfaced one.**
- [ ] 10.6 **Section 7 — Comparative Weight Diff Across Published Gemma 4 E4B Abliterations.** Cite `results/weight_diffs/` and `results/figures/` (overlay chart, cross-method cosine, refusal-direction vs singular-vector). **No MoE / router / expert content.**
- [ ] 10.7 Section 8 (Discussion + Course Connections) — over-parametrization, matrix perturbation (Hoffman-Wielandt), NTK.
- [ ] 10.8 Section 9 (Conclusion + Ethics).
- [ ] 10.9 Integrate all sections; compile.
- [ ] 10.10 Slide deck: hiking opener, 3+ key figures, course-connections slide.
- [ ] 10.11 `paper/sources.md` — every numeric claim mapped to source file path + commit hash.
- [ ] 10.12 Self-critique pass for consistency / formatting.
- [ ] 10.13 Final review; write `READY_FOR_SUBMISSION.md`.
- [ ] 10.14 Push final commits on `agent/writeup`.

## 11. Cleanup / Hand-off

- [ ] 11.1 Operator reviews `READY_FOR_SUBMISSION.md`.
- [ ] 11.2 Operator merges `agent/writeup` to `main`.
- [ ] 11.3 Optionally remove worktrees with `git worktree remove ../gb-<name>`.
- [ ] 11.4 Archive this change once project complete: `openspec archive gemma-only-execution-plan --skip-specs --yes`.
