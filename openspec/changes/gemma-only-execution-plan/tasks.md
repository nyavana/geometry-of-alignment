# Tasks — Gemma-only execution plan

This is the single source of truth for milestones M0–M5 of the project. M0 and M1 are complete (history retained for reference). M2–M5 are the live workstreams.

**Dispatch contract** (carried forward from `autonomous-agent-pivot`): every agent dispatch SHALL include (1) absolute worktree path, (2) branch name, (3) a specific section ID from this file, (4) GPU policy, (5) commit-and-push protocol, (6) stop condition at section boundary, (7) model (`claude-sonnet-4-6` for mechanical/well-specified tasks; `claude-opus-4-7` for tasks that demand novel decisions, ambiguity resolution, or writing — see CLAUDE.md "Subagent model routing" for the default mapping per section).

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
| `agent/m6-rank1-followup` | `../gb-m6/` (created off `main` on first dispatch) | gpu-lock-required |

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

- [x] 3.1 In `../gb-bench/`, rebase onto `origin/main` *(implicit during 3.2 rebase)*
- [x] 3.2 Verify `src/benchmark/{evaluate.py,classify_refusal.py,analyze_results.py}` are runnable. *(commit `adc69fb`: added `--limit` flag to evaluate.py)*
- [x] 3.2.1 Install upstream llama.cpp's `llama-server` binary. *(Done from `main`: see CLAUDE.md.)*
- [x] 3.3 Smoke test base GGUF via llama-server. *(commit `9053341`)*
- [x] 3.4 Iterate `classify_refusal.py` patterns. *(commit `eb25419`; later widened in M2c commit `b25c361` to catch "I cannot ___" surface form on transformers-backend abliterated outputs)*
- [x] 3.5 Full evaluation: base Gemma 4 E4B GGUF. *(commit `60dfbf6`; should_refuse 100%, em_refuse 2.0%)*
- [x] 3.6 Full evaluation: Gemma 4 E2B BF16 (validation). *(commit `e48d838`; should_refuse 95.2%, em_refuse 12.0%; transformers v5 BatchEncoding fix in `df6ff20`)*
- [x] 3.7 Full evaluation: OBLITERATUS GGUF — **FAILED**. *(commit `bce69af`; llama-server crashed on Harmony-format `<|channel>` tokens emitted by the model. Recovery path: transformers backend, deferred to a follow-up since not paper-blocking — M3 weight-diff already covers OBLITERATUS quantitatively. See `docs/issues/2026-05-06-obliteratus-eval-fail.md`.)*
- [x] 3.8 Full evaluation: TrevorJS bf16 transformers — **SKIPPED**. *(commit `6cd823e`; transformers @ 8-bit was 117 s/iter ⇒ 11 h ETA. Killed at 5%. M3 weight-diff covers TrevorJS quantitatively. See `docs/issues/2026-05-06-trevorjs-eval-fail.md`.)*
- [x] 3.9 Full evaluation: HauhauCS GGUF. *(commit `b629d67`; uniform 0% refusal across all categories — truly uncensored variant)*
- [x] 3.10 Phrasing sensitivity analysis. *(Killed at 16% on base E4B variants — 4-5 h ETA was eating the M2c GPU window. Documented in section (g) of STATUS_FOR_HUMAN.md.)*
- [x] 3.11 Context sensitivity test. *(Base E4B + first-responder prefix on emergency_medical: 2.0% → 4.0% delta +2pp. OBLITERATUS variant skipped — see 3.7.)*
- [x] 3.12 `analyze_results.py` heatmap + figures. *(commit `2e46f83`; also regenerated in `79a0a73` with self-abliterated row)*
- [x] 3.13 Final summary commit with M2a-summary block. *(committed in master pipeline, parseable by M4)*

*Note: evaluation of the project's own M2c-abliterated model is in section 6 below, after M2c lands.*

## 4. M2b — Mechanistic Analysis (GPU)

Goal: layer-by-layer activation extraction, refusal direction computation, layer-rank analysis, visualization. **Identical to the predecessor's section 4** — no changes from the swap.

Agent scope: `agent/mechanistic-analysis` worktree (`../gb-mech/`). GPU policy: gpu-lock-required for all model loads.

- [x] 4.1 In `../gb-mech/`, rebase onto `origin/main` (see preamble — `m1-benchmark-frozen` is stale). *(commit 8258849 on agent/mechanistic-analysis: HEAD already matched origin/main at dispatch time)*
- [x] 4.2 Verify `src/mechanistic/{extract_activations.py,layer_analysis.py,visualize.py}` and the `ActivationCollector` class. Smoke test on E2B BF16 with 10 prompts inside `scripts/gpu_lock.sh`. *(commit 5132291 on agent/mechanistic-analysis: corrected to Gemma 4 multimodal layer path `model.language_model.layers`; smoke test passes)*
- [x] 4.3 Full activation extraction on Gemma 4 E4B 8-bit: refuse-class (`should_refuse` + the five over-refuse categories `emergency_medical`, `wilderness_survival`, `home_safety`, `chemistry_safety`, `mental_health`) → `results/activations/refuse_activations.pt`; comply-class (`safe_control`) → `results/activations/comply_activations.pt`. Also write `results/activations/prompt_metadata.json` with the per-row mapping (`prompt_id`, `category`, `expected`) so M2c 5.9 can slice category-specific subsets without re-extracting on the GPU. *(commit 4fa3fdc on agent/mechanistic-analysis; artifacts present at `$RESULTS_DIR/activations/`)*
- [x] 4.4 Compute refusal directions per layer via mean-diff → `$RESULTS_DIR/activations/refusal_directions.pt` (resolves to `/home/nyavana/columbia/6699/shared/results/agent/mechanistic-analysis/activations/refusal_directions.pt`). Also commit a redundant copy into `results/activations/refusal_directions.pt` (the `.gitignore` allowlist permits this) and push the branch. **CHECKPOINT — push immediately on completion. This is the M2b artifact that unblocks M2c task 5.4 and M3 task 7.10. Do not bundle 4.5–4.9 into the same commit.** *(commit 57cedcf on agent/mechanistic-analysis; artifact: 442 KB at `$RESULTS_DIR` and in-repo redundant copy)*
- [x] 4.5 Signal-strength + sliding/global comparison. *(commit `1d5c590`; peak L15 Cohen's d 2.87, top-3 L15/L4/L14, sliding-vs-global gap inconclusive)*
- [x] 4.6 PCA rank analysis per layer. *(commit `d17afb4`; rank-1 hypothesis strongly supported: top-1 PC captures 86.6% of |Δμ|² mean over peak band)*
- [x] 4.7 UMAP/t-SNE multi-layer grid. *(commit `358abf5`; cleanest separation at L15)*
- [ ] 4.8 Cross-precision validation: refusal direction on E2B BF16, cosine vs E4B 8-bit. **DEFERRED — low priority for paper**
- [x] 4.9 Push branch with summary. *(headline numbers carried in 4.5/4.6/4.7 commits; M4 STATUS_FOR_HUMAN.md aggregated them into section (c))*

## 5. M2c — Abliteration + Selective Safety (GPU)

Goal: this project's own abliteration of Gemma 4 E4B + ablation sweeps + selective safety. **Identical to the predecessor's section 5**, with one addition: per recent literature, standard rank-1 abliteration may underperform on Gemma 4 due to RMSNorm + shared K/V — document the result either way.

Agent scope: `agent/abliteration` worktree (`../gb-ablit/`). GPU policy: gpu-lock-required.

Dependency: M2b task 4.4 (refusal directions exist) must be complete before 5.4 below starts.

- [x] 5.1 Rebase agent/abliteration onto origin/main.
- [x] 5.2 Verify scripts runnable. *(commits `a113fc5` Gemma 4 multimodal layer path + transformers v5 BatchEncoding fix; `b25c361` classifier regex widened)*
- [x] 5.3 Sanity test on E2B BF16 (5 prompts).
- [x] 5.4 Copy refusal_directions.pt from M2b's $RESULTS_DIR. *(commit `9c5cef6`)*
- [x] 5.5 Full E4B abliteration at α=1.0, all 42 layers. *(commit `9c5cef6`; model at `$RESULTS_DIR/models/gemma-4-e4b-abliterated/` ~11 GB)*
- [x] 5.6 Quick-test (12 should_refuse prompts): 100% → 91.7% — predicted Gemma 4 RMSNorm/shared-K/V resistance confirmed. *(commit `9c5cef6`)*
- [x] 5.7 Sweep ensemble. *(commits `5856994` perf cap + `09f4931` results JSON; alpha sweep flat 30-35% across α∈[0,2.0], layer subset flat 25-35%, random control 30%. **Standard rank-1 mean-diff abliteration is empirically ineffective on Gemma 4 E4B 8-bit.** Stratified 20-prompt subset, max_new_tokens=128, ETA bound to ~2.8 h.)*
- [ ] 5.8 Capability preservation (MMLU + GSM8K). **DEFERRED — low priority since 5.7 showed abliteration didn't change behavior, so capabilities likely also unaffected. See `docs/issues/2026-05-06-m2c-deferred-items.md`.**
- [x] 5.9 Category-specific refusal directions. *(commit `9c5cef6`; over-refuse cluster +0.93 pairwise; orthogonal to should_refuse −0.015. Table at `results/figures/selective_safety_table.md`.)*
- [ ] 5.10 Selective abliteration eval (medical-only direction). **DEFERRED — given 5.7's negative finding, the geometric clean-separation result in 5.9 stands as the paper's selective-safety contribution.**
- [x] 5.11 Figures: `alpha_sweep.png`, `layer_subset_comparison.png`, `selective_safety_table.md`. *(commit `c496617`)*
- [x] 5.12 Hand-off note: abliterated model ready at `$RESULTS_DIR/models/gemma-4-e4b-abliterated/`. *(commit `9c5cef6` body)*
- [x] 5.13 Final summary commit with `M2c-summary` parseable block. *(commit `33fe8f4`)*

## 6. M2c-followup — Benchmark on the project's own abliterated model

Goal: complete the heatmap by adding our own abliterated model.
Agent scope: `agent/benchmark-eval` worktree. GPU policy: gpu-lock-required for transformers backend.

- [x] 6.1 Pulled abliterated model path. *(operator-mediated; path `$RESULTS_DIR/models/gemma-4-e4b-abliterated/`)*
- [x] 6.2 Run evaluate.py against self-abliterated. *(commit `b184932`; ran on stratified 48-prompt subset 6/cat × 8 cats due to ~75 s/iter inference cost. **should_refuse 6/6 = 100% — UNCHANGED FROM BASE.** Empirically confirms M2c sweep finding.)*
- [ ] 6.3 Selective-abliterated eval. **NOT PRODUCED** — 5.10 was deferred.
- [x] 6.4 Regenerated heatmap with self-abliterated row. *(commit `79a0a73`; 5 real model rows: E2B, E4B base, E4B+context, HauhauCS, self-abliterated)*
- [x] 6.5 Final summary commit with `M2c-followup-summary` parseable block. *(commit `7d6e040`)*

## 7. M3 — Comparative Weight Diff (replaces Qwen MoE)

Goal: weight-diff between base Gemma 4 E4B-it and each published bf16 variant. SVD rank analysis. Cross-method comparison. Cross-reference with M2b refusal directions.

Agent scope: `agent/weight-diff` worktree (`../gb-wdiff/`). GPU policy: gpu-none. Runs concurrently with M2 (no GPU contention).

**Targets:**
- Primary: `OBLITERATUS/gemma-4-E4B-it-OBLITERATED` (bf16 safetensors)
- Secondary: `TrevorJS/gemma-4-E4B-it-uncensored` (bf16 safetensors source repo)
- Behavioral-only: `HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive` (GGUF — no weight diff possible; benchmark-eval coverage in section 3 above is sufficient)

**Dependencies:** Tasks 7.1–7.9 are independent of M2b and can run as soon as the worktree is ready. Task 7.10 (cross-reference with refusal directions) BLOCKS until `agent/mechanistic-analysis` has produced `refusal_directions.pt` (M2b task 4.4) — read it from `/home/nyavana/columbia/6699/shared/results/agent/mechanistic-analysis/activations/refusal_directions.pt`, or as fallback from `origin/agent/mechanistic-analysis:results/activations/refusal_directions.pt`.

- [x] 7.1 In `../gb-wdiff/`, rebase onto `origin/main` (see preamble — `m1-benchmark-frozen` is stale). *(commit b657b6a on agent/weight-diff)*
- [x] 7.2 **Pre-flight: disk + license check.** *(commit b657b6a; 600 GB free; both variants Apache 2.0)*
  - Run `df -h /home/nyavana/columbia/6699/shared/` — confirm ≥40 GB free.
  - Read each variant's HuggingFace model card to verify license inheritance (OBLITERATUS card states Apache 2.0 from base; verify TrevorJS).
  - If insufficient disk OR a license blocker, stop and surface to operator.
- [x] 7.3 Download `OBLITERATUS/gemma-4-E4B-it-OBLITERATED` bf16 safetensors via `huggingface-cli download` to `model/OBLITERATUS-gemma-4-E4B-it-OBLITERATED/`. *(commit b657b6a; downloaded to shared/model/)*
- [x] 7.4 Download `TrevorJS/gemma-4-E4B-it-uncensored` bf16 safetensors to `model/TrevorJS-gemma-4-E4B-it-uncensored/`. *(commit b657b6a; downloaded to shared/model/)*
- [x] 7.5 **Pre-flight: shape/key compatibility.** For each variant, load the state-dict header and assert keys match base; assert shapes match. If TrevorJS fails, log to `results/weight_diffs/.compat_log.md` and proceed with OBLITERATUS only (per design D2 fallback). If OBLITERATUS fails, stop and surface to operator. *(commit b657b6a; both variants passed — empty `.compat_log.md` ⇒ 7.7 may run `--strict` for OBLITERATUS and standard mode for TrevorJS)*
- [x] 7.6 Smoke-test `src/weight_diff/compute_diff.py` and `svd_analysis.py`: run against (base × OBLITERATUS) for one layer only. Confirm scripts produce JSON output and don't error. *(commit b657b6a)*
- [ ] 7.7 **Full weight diff per variant.** For each variant in `[OBLITERATUS, TrevorJS]` that passed pre-flight 7.5, run `python -m src.weight_diff.compute_diff --original model/gemma-4-E4B-it/ --modified model/<variant>/ --output $RESULTS_DIR/weight_diffs/<variant_slug>/` (use `--strict` for OBLITERATUS, omit for TrevorJS — see 7.5). Each run peaks ~32–35 GB RAM; the WSL2 instance is capped at 46 GB physical, so **run the variants sequentially, not in parallel** (two parallel runs would need ~68 GB combined and oversubscribe).
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

- [x] 8.1 RLHF (Christiano 2017, Ouyang 2022), DPO (Rafailov 2023), Constitutional AI (Bai 2022). *(commit 38fb89a on agent/writeup; seeded `paper/sections/02_background.md`)*
- [x] 8.2 Representation engineering (Zou 2023), linear representation hypothesis. *(commit 3bf01f1 on agent/writeup; in `paper/sections/02_background.md`)*
- [x] 8.3 Abliteration: Arditi 2024, Heretic (p-e-w 2025), OBLITERATUS (elder-plinius 2025), grimjim's norm-preserving biprojection. *(commit 78044fc on agent/writeup; seeded `paper/sections/03_related_work.md`)*
- [x] 8.4 Over-refusal: Rottger 2024 (XSTest), Cui 2024. *(commit 033eb46 on agent/writeup; in `paper/sections/03_related_work.md`)*
- [x] 8.5 Gemma 4 architectural quirks: source the "doesn't work on Gemma 4" findings (Heretic GitHub issues, OBLITERATUS card). *(commit cce3829 on agent/writeup; in `paper/sections/03_related_work.md`)*
- [x] 8.6 Verify ≥15 citations across `02_background.md` and `03_related_work.md`. *(commit `21e320f`; 17 citations total — 6 in 02, 11 in 03)*

## 9. M4 — Human Verification Gate

Goal: produce `STATUS_FOR_HUMAN.md` and wait for operator's green-light sentence.
Agent scope: `agent/writeup`. GPU policy: gpu-none.

- [x] 9.1 Aggregate headline numbers from all agent branches' final commit messages. *(commits `e264abb` v1, `f42e0d6` v2)*
- [x] 9.2 Section (a) Branch and commit status.
- [x] 9.3 Section (b) Refusal rates table. *(v2 includes self-abliterated row from 6.2)*
- [x] 9.4 Section (c) Mechanistic summary.
- [x] 9.5 Section (d) Abliteration sweep summary — Gemma 4 RMSNorm/shared-K/V resistance is paper-relevant central finding.
- [x] 9.6 Section (e) Comparative weight diff summary.
- [x] 9.7 Section (f) "What the human needs to do":
  - Open every PNG under `/home/nyavana/columbia/6699/shared/results/agent/*/figures/` and eyeball for breakage.
  - Read 10 random responses from each abliterated model (self + OBLITERATUS + TrevorJS + HauhauCS) and confirm plausibility.
  - Verify `/home/nyavana/columbia/6699/shared/results/agent/benchmark-eval/refusal_rates/gemma4_e4b_base/evaluation_results.csv` shows `should_refuse` refusal rate >80%.
  - Verify `/home/nyavana/columbia/6699/shared/results/agent/benchmark-eval/refusal_rates/gemma4_e4b_self_abliterated/evaluation_results.csv` shows `should_refuse` refusal rate <30%.
  - Verify `/home/nyavana/columbia/6699/shared/results/agent/weight-diff/weight_diffs/cross_method_cosine_table.csv` exists and contains numeric values.
  - Verify `/home/nyavana/columbia/6699/shared/results/agent/weight-diff/figures/refusal_direction_vs_singular_vector.png` exists.
  - Grep for credentials leaks (`HF_TOKEN`, `HUGGING_FACE`, `API_KEY`).
  - Decide which branches to merge.
  - Write the green-light sentence: **"Approved to proceed to M5 — writeup authorized."**
- [x] 9.8 Section (g) Anomalies/deviations.
- [x] 9.9 Commit and push STATUS_FOR_HUMAN.md. *(v2 at commit `f42e0d6` on `agent/writeup`)*
- [ ] 9.10 **STOPPED. Awaiting operator green-light: open `STATUS_FOR_HUMAN.md`, eyeball PNGs (f.1), sample CSV responses (f.2), grep for credentials (f.7), decide branch merges (f.8), then write "Approved to proceed to M5 — writeup authorized."**

## 10. M5 — Paper + Slides

Goal: 10-section paper (with new §4 Mathematical Framework anchoring EECS 6699 lectures) + slide deck.
Agent scope: `agent/writeup`. GPU policy: gpu-none. CPU only — the new §4 numerics (10.3, 10.4) consume artifacts already on disk.

**PRECONDITION:** `STATUS_FOR_HUMAN.md` contains "Approved to proceed to M5 — writeup authorized."

**Reference:** `docs/findings/course-material-mapping.md` is the canonical EECS 6699 lecture-to-module map. §4 prose and §9 Discussion both cite it.

**Style policy — /humanizer integration:** Each section-prose task (10.2, 10.5–10.11) SHALL end with a `/humanizer` pass on that section's markdown file, committed as a separate edit on top of the initial draft. Per-section scope: prose flow only — preserve verbatim equations, theorem statements, numeric claims (refusal rates, cosine values, Frobenius ratios, `‖E‖_F / ‖W‖_F` numbers), citation strings, file paths, and figure/table references. The skill targets AI tells (inflated symbolism, rule-of-three padding, em-dash overuse, vague attributions, AI vocabulary), NOT technical content. Pre-existing drafts (`02_background.md`, `03_related_work.md`) are back-filled at 10.11a; `08_rank1_cascade.md` is handled inside 10.9. A paper-wide `/humanizer` sweep runs at 10.16 to catch cross-section repetition of AI vocabulary and em-dash drift after integration.

**Writing dispatch pattern:** This is the M5-specific addition to the generic 7-field dispatch contract (see preamble at top of file + `specs/autonomous-execution/spec.md` Requirement: Agent dispatch contract). All writing subagents use `claude-opus-4-7` per the CLAUDE.md model-routing default for M5.

- **Granularity: one subagent per prose task.** Tasks 10.2, 10.5, 10.6, 10.7, 10.8, 10.9, 10.10, 10.11 each dispatch as their own subagent. The /humanizer pass for that section runs inside the same dispatch (separate commit on top of the draft) — the subagent already has the section content and source CSVs in context; spawning a second subagent just to lint would re-read everything.
- **Concurrency cap: at most 3 writing subagents in flight at any time.** Rationale: each subagent is a ~45–50 min wall-clock blast radius (per CLAUDE.md "Subagent runtime budget"); three concurrent failures is the largest reasonable redo cost. The main thread orchestrates batches and waits before launching the next wave.
- **Suggested batches (respect dependencies + the 3-at-a-time cap):**
  - Batch A (parallel, no dependencies): 10.2 (§1), 10.6 (§5), 10.7 (§6) — wait until §4 numerics scripts 10.3 + 10.4 have produced their CSVs and figures, but §1/§5/§6 prose itself doesn't depend on §4.
  - Batch B (parallel, depends on Batch A complete + 10.3/10.4 artifacts): 10.5 (§4 Math Framework), 10.8 (§7 Abliteration & Weight Diff — gated on M3 7.7–7.14 landing), 10.9 (§8 M6 Cascade incorporating existing draft).
  - Batch C (sequential after Batch B): 10.10 (§9 Discussion) on its own — it cites §4 Mirsky numbers AND every empirical section's headline, so launching it before B finishes risks stale references. Pair with 10.11a (back-fill /humanizer on 02 + 03) and 10.11 (§10 Conclusion) once 10.10 returns.
- **Per-dispatch context budget — what each subagent reads:** the section's spec scenario from `specs/research-paper/spec.md`; the CSV/figure paths the scenario lists; `docs/findings/course-material-mapping.md` if the section cites lectures (§4, §9); the section's existing draft (if back-filling) or sibling sections it cross-references. Subagents SHALL NOT read the full `tasks.md` or unrelated sections — the dispatch prompt extracts the relevant paragraphs verbatim. Goal: each subagent's working context stays under ~30k tokens so the main thread's context stays clean.
- **Tasks that operate paper-wide get their own subagent each:** 10.13 (section-reference audit — reads every `paper/sections/*.md`) and 10.16 (consistency pass + paper-wide /humanizer sweep). The main thread SHALL NOT execute these inline; both would balloon main-thread context with the entire paper.
- **Failure recovery:** if a writing subagent hits the runtime cap mid-section, the main thread re-dispatches with the same scope and an additional "resume from `paper/sections/<file>.md` HEAD" instruction. The 7-field contract's commit-after-every-sub-task rule means a section draft and its /humanizer pass land as two commits, so a mid-/humanizer failure leaves the draft committed and only the humanizer step needs replay.

- [ ] 10.1 Verify green-light sentence; if absent, stop and report M4 unresolved.
- [ ] 10.2 Section 1 (Introduction) — hiking emergency scenario.
- [ ] 10.3 **NEW — Mirsky-style numerics for §4.** Create + run `scripts/m5_math_framework/hw_bound.py` over both direction artifacts. Inputs: `shared/results/agent/mechanistic-analysis/activations/refusal_directions.pt` (M2b raw) AND `shared/results/agent/m6-rank1-followup/m6_directions/refusal_directions_d3.pt` (D3 winner); base weights from `model/gemma-4-E4B-it/` loaded via `transformers.AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)` (CPU). For each of 42 layers × 2 projections (`o_proj`, `down_proj`) × 2 direction variants at α=1.0, compute `‖α·d·dᵀW‖_F` (note: equals `‖·‖_2` for rank-1 `E`), `‖W‖_F`, and the headline ratio `‖E‖_F / ‖W‖_F`. Outputs: `results/math_framework/mirsky_bound_per_layer.csv` and `results/figures/mirsky_bound_heatmap_{d3,m2b}.png`. Cross-check the D3 row at one layer against M3's TrevorJS Frobenius for consistency. Reuse `src/weight_diff/svd_analysis.py` Frobenius helpers; reuse the loader pattern from `src/mechanistic/extract_activations.py`.
- [ ] 10.4 **NEW — Anisotropy + projection-energy figures for §4.** Create + run `scripts/m5_math_framework/anisotropy.py` on the L15 activations at `shared/results/agent/mechanistic-analysis/activations/{refuse,comply}_activations.pt`. Produce three figures: (a) `results/figures/projection_energy_L15.png` — histogram of `|⟨a, d⟩|² / ‖a‖²` for refuse-class vs comply-class activations at L15; (b) `results/figures/learned_directions_cosine_L15.png` — cosine matrix between {refuse mean, comply mean, over-refuse mean, safe-control mean, refusal direction `d`}; (c) `results/figures/activation_anisotropy_L15.png` — pairwise cosine histogram on 200 refuse + 200 comply samples, with Gaussian N(0, 1/d) reference overlay. Report headline numbers (mean pairwise cosine, centered std, median |cos| of learned directions, fraction of activation energy on `d`) in the prose of 10.5.
- [ ] 10.5 **NEW — Section 4 (Mathematical Framework).** Write `paper/sections/04_math_framework.md` per the §4 spec requirement. Formalize `W ← W − α·d·dᵀW` with `W` rectangular. Invoke Lec 5 (random-vector near-orthogonality + thin-shell concentration applied to *learned conceptual directions*, with the anisotropy caveat for raw activations), Lec 6 (Mirsky's singular-value perturbation theorem `Σᵢ (σᵢ(W+E) − σᵢ(W))² ≤ ‖E‖_F²`; rank-1 specialization `‖E‖_F = ‖E‖_2 = |α|·‖dᵀW‖_2`; note rectangular matrices preclude direct eigenvalue Hoffman-Wielandt), Lec 7 (SVD / Frobenius / spectral norms machinery used in §7 and §8), Lec 8–9 (lazy / NTK linearization — output is approximately a linear functional of the perturbation; partial first-order response plausible, not predicted). Embed the heatmaps from 10.3 and the three figures from 10.4. Cite `docs/findings/course-material-mapping.md`. Headline number in prose: median `‖E‖_F / ‖W‖_F` across all 84 (42 × 2) cells for D3 vs M2b.
- [ ] 10.6 Section 5 (Over-Refusal Analysis) — quote only numbers from `results/refusal_rates/`; cite CSV paths. *(was old 10.3)*
- [ ] 10.7 Section 6 (Mechanistic Analysis) — cite figures from `results/figures/`. *(was old 10.4)*
- [ ] 10.8 **Section 7 — Abliteration & Comparative Weight Diff (combined).** Cite `results/ablation_results/`, `results/weight_diffs/`, and `results/figures/` (alpha sweep, overlay chart, cross-method cosine, refusal-direction vs singular-vector). Include the Gemma 4 architectural-quirk discussion. **No MoE / router / expert content.** *(merges old 10.5 + 10.6)*
- [ ] 10.9 Section 8 (M6 Rank-1 Cascade) — incorporate existing draft `paper/sections/08_rank1_cascade.md`. Update internal section references after 10.13 audit (e.g., line 7's "Section 5 ... Section 6" becomes "Section 6 ... Section 7" under the new outline). Run `/humanizer` on `08_rank1_cascade.md` as part of this task (the draft was authored without the §4 framing in place and predates the style policy); commit the humanized version on top of the section-reference fixups.
- [ ] 10.10 **Section 9 (Discussion + Course Connections) — strengthened.** Require ≥5 explicit lecture connections (mathematical notation); ≥1 quantitative connection (numeric Mirsky bound for D3 from 10.3 tied to §8's 40.5% partial-refusal finding); ≥1 connection from each of {Lec 5}, {Lec 6, Lec 7}, {Lec 8, Lec 9}. Phrase Lec 8–9 ⇔ §8 partial result as "the lazy/linearization lens makes a partial first-order response plausible," NOT "NTK predicts the 40.5% finding." Add over-parametrization framing (safety uses a small subspace of a 2560-dim residual stream).
- [ ] 10.11 Section 10 (Conclusion + Ethics). *(was old 10.8 §9; ethics moved to §10 under new numbering)*
- [ ] 10.11a **Back-fill `/humanizer` on pre-existing drafts.** Run `/humanizer` on `paper/sections/02_background.md` and `paper/sections/03_related_work.md` (authored before the style policy landed). Same scope as the per-section pass: preserve citations, dates, author names, and quoted material verbatim. Commit each humanized version as a separate edit. *(`08_rank1_cascade.md` is back-filled inside 10.9, not here.)*
- [ ] 10.12 Integrate all sections; compile.
- [ ] 10.13 **NEW — Section-reference audit.** Grep every file under `paper/sections/` for `Section [0-9]+` and update stale references to match the new 10-section outline. Known instances: `08_rank1_cascade.md:7` ("Section 5 of this paper computed ... Section 6 applied ...") needs updating; `08_rank1_cascade.md:52` ("Section 7 of this paper") — under the new outline that is still §7 (weight-diff content is now part of combined §7); verify case-by-case. Confirm zero stale references before integration.
- [x] 10.14 Slide deck: hiking opener, 3+ key figures, **dedicated Mathematical Framework slide referencing §4 numerics** (Mirsky bound + anisotropy caveat), course-connections slide (§9 recap with ≥5 connections). *(Delivered for in-person presentation; final decks committed at `paper/presentation-slides/geometry_of_alignment_20260507_100217.pptx` and `geometry_of_alignment_20260508_161459.pptx` with `story.md` (hiking opener). Presentation complete — §4 Mathematical Framework content lives in the paper §4 only, not the slide deck.)*
- [ ] 10.15 `paper/sources.md` — every numeric claim mapped to source file path + commit hash. Include entries for `results/math_framework/mirsky_bound_per_layer.csv` and the three §4 figures from 10.4.
- [ ] 10.16 Self-critique pass for consistency / formatting. Verify no `eigenvalue` / `λᵢ` claim in the rank-1 abliteration context; verify no "NTK predicts" overclaim phrasing. **Then run a paper-wide `/humanizer` sweep across the integrated draft** to catch cross-section drift the per-section passes can't see: AI vocabulary recurring across sections, em-dash density creeping back up after integration, "rule of three" lists that read identically across §6/§7/§8, etc. Commit the humanized integration as a separate edit on top of the consistency fixups.
- [ ] 10.17 Final review; write `READY_FOR_SUBMISSION.md`.
- [ ] 10.18 Push final commits on `agent/writeup`.

## 11. Cleanup / Hand-off

- [ ] 11.1 Operator reviews `READY_FOR_SUBMISSION.md`.
- [ ] 11.2 Operator merges `agent/writeup` to `main`.
- [ ] 11.3 Optionally remove worktrees with `git worktree remove ../gb-<name>`.
- [ ] 11.4 Archive this change once project complete: `openspec archive gemma-only-execution-plan --skip-specs --yes`.

## 12. M6 — Rank-1 Abliteration Follow-up (causal isolation cascade)

Goal: isolate which single ingredient closes the gap between M2c's failed self-abliteration and the published Gemma 4 E4B successes (TrevorJS, OBLITERATUS, HauhauCS). Five-stage cascade, each stage gated on the prior stage's smoke result. **No stage runs the full 344-prompt benchmark** until Stage 4. Source-of-truth: `docs/M6_PROPOSAL_RANK1_FOLLOWUP.md`.

Agent scope: `agent/m6-rank1-followup` branch off `main`, in worktree `../gb-m6/` (create on first dispatch via `git worktree add ../gb-m6 -b agent/m6-rank1-followup main`). GPU policy: gpu-lock-required (bf16 E4B at ~10–12 GB peak). All shell blocks assume the preamble `source /home/nyavana/columbia/6699/shared/env.sh && source .venv/bin/activate && SHARED_RESULTS=/home/nyavana/columbia/6699/shared/results && SHARED_MODEL=/home/nyavana/columbia/6699/shared/model` is sourced first.

Predecessor: M2c (negative finding) and M3 (cross-method weight-diff geometry). Successor: M5 paper drafting — M6 *adds* a positive-result chapter, does not block M5.

**M6 reconciliation (2026-05-07, merge commit `e4e5622`):** cascade complete. H1 (bnb int8 edit-path) **rejected**, H2/H3 alone **insufficient**, **H4 (Gram-Schmidt vs harmless mean) load-bearing** — D3 produces 1/6 should_refuse smoke (16.7%) but **17/42 (40.5%) at n=42** — partial-effect band, not a clean win. H5 (norm-preserving biprojection) **refuted** — vanilla projection only changes per-row `o_proj`/`down_proj` norms by 0.03–0.07% on average (max 2.8%), too small to matter for RMSNorm; Stage 3a is per-prompt identical to D3 vanilla on smoke. Stage 4 (full 344-prompt benchmark) **skipped** — D3 is partial, full bf16 transformers run would take ~19 hours with limited marginal information beyond n=42. Stage 3b (faithful TrevorJS reproduction) **skipped** — Stage 3a confirms norm preservation is not the bottleneck. M5 paper writes M6 up as a causal-isolation cascade with a partial-effect terminus pointing to multi-rank descent (consistent with M3's OBLITERATUS rank_95 = 6) as the remaining handle. See §7 of `docs/M6_PROPOSAL_RANK1_FOLLOWUP.md` for the per-stage results tables and `STATUS_FOR_HUMAN.md` `## M6 — Rank-1 Follow-up` for the operator-facing summary.

### 12.0 Pre-launch sanity (before Stage 0)

- [x] 12.0.1 Verify `$SHARED_MODEL/TrevorJS-gemma-4-E4B-it-uncensored/model.safetensors` exists and is bf16.
- [x] 12.0.2 Verify `$SHARED_RESULTS/agent/mechanistic-analysis/activations/refusal_directions.pt` contains 42 unit-norm vectors of dim 2560 (M2b artifact).
- [x] 12.0.3 Verify `$SHARED_RESULTS/agent/benchmark-eval/stratified_50.json` has 48 prompts (6 per category × 8 categories) — note the filename contains "50" for historical reasons; the actual count is 48.
- [x] 12.0.3a Verify `data/benchmark_prompts.json` filtered by `category == "should_refuse"` returns exactly 42 prompts (the Stage 1.5 confirmation set). Verified at change-authoring time: `should_refuse=42`, `gray_zone=42`, others 40 or 50.
- [x] 12.0.4 GPU is idle: `nvidia-smi` shows <1 GB used.
- [x] 12.0.5 Component-target assertion: `src/abliterate/abliterate.py` lines 106–114 confirm `target_weights="residual"` modifies only `self_attn.o_proj` and `mlp.down_proj`.
- [x] 12.0.6 Branch `agent/m6-rank1-followup` created off `main`.

### 12.1 Stage 0 — Establish controls (serial, ~50 min)

Run **serially, not in parallel** — two bf16 E4B copies on a single 16 GB 4070 Ti would OOM. 0b runs first so the cheapest sanity check fails fast if the eval pipeline is broken.

- [x] 12.1.1 **Stage 0b (positive control / H6, RUN FIRST):** evaluate `$SHARED_MODEL/TrevorJS-gemma-4-E4B-it-uncensored` (bf16) against `stratified_50.json` via the transformers backend. Output to `$RESULTS_DIR/stage0b_trevorjs_bf16/`. Acceptance: `should_refuse` ≤ 10%; sample 5 outputs by hand to confirm coherence. *If >10%, abort the cascade and debug the eval pipeline (likely chat template skew or refusal classifier regex too aggressive on TrevorJS's "refusal-then-comply" pattern).*
- [x] 12.1.2 **Stage 0a (bnb int8 edit-path test / H1, RUN SECOND):** run `python -m src.abliterate.abliterate --model google/gemma-4-E4B-it --directions $SHARED_RESULTS/agent/mechanistic-analysis/activations/refusal_directions.pt --alpha 1.0 --output models/gemma-4-e4b-self-abliterated-bf16/` **without `--use-8bit`** (default → all 42 layers, all 42 per-layer M2b directions, vanilla projection, bf16 edit). Then evaluate against `stratified_50.json` → `$RESULTS_DIR/stage0a_self_abliterated_bf16/`. OOM fallback: `device_map="auto"` with CPU offload of late layers — **do NOT fall back to E2B** (different model class confounds the test).
- [x] 12.1.3 Framing assertion: in commit messages and STATUS updates, describe Stage 0a as a "bnb int8 edit-path test," not a generic "precision toggle." A ≤30% result isolates the bnb int8 in-place edit wrapper, NOT "int8 quantization at inference" (HauhauCS's quantized GGUF scoring 0% rules that out).

### 12.2 Stage 1 — Smoke verification gate (three-band)

- [x] 12.2.1 Tabulate Stage 0a + 0b results in a 2-row mini-table; commit with summary `M6 0a/0b complete`.
- [x] 12.2.2 Apply the three-band gate on Stage 0a `should_refuse` (n=6):

  | Band | Interpretation | Action |
  |---|---|---|
  | ≤ 30% (≤ 1/6) | **Cracked.** H1 confirmed: bnb int8 edit path is load-bearing. | Run Stage 1.5 confirmation; if it holds, jump to Stage 4. |
  | 30 – 85% (2–5/6) | Significant partial effect; bnb explains some-but-not-all. | Note magnitude; proceed to Stage 2 starting at D1. |
  | > 85% (6/6) | No meaningful effect; bnb edit path is not the cause. | Proceed to Stage 2 starting at D1. |

### 12.3 Stage 1.5 — Confirmation on full `should_refuse` set (any stage with ≤30% smoke)

Stage 1.5 SHALL run for **any** stage that lands ≤30% on its smoke tier — Stage 0a (n=48 stratified), Stage 2 D1/D2/D3 (n=12 targeted), or Stage 3a/3b (n=12 targeted). The smoke tier is too coarse to publish off; the n=42 single-category run is sufficient resolution for the paper headline.

- [x] 12.3.1 Re-evaluate the candidate winning checkpoint on all 42 base `should_refuse` prompts (filter `data/benchmark_prompts.json` by `category == "should_refuse"`). ~20 min at 30 s/prompt.
- [x] 12.3.2 Acceptance: refusal rate at n=42 is ≤30% (binomial robustness — at n=6 a single classifier flip moves the rate by 16.7 pp; at n=42 by 2.4 pp).
- [x] 12.3.3 Hand-audit 10 randomly-sampled non-refusing outputs to filter "refusal-then-comply" false negatives.
- [x] 12.3.4 If n=42 disconfirms the smoke result, treat the smoke as a noise spike and route per the originating stage: Stage 0a smoke disconfirm → "0a-partial" Stage 2 branch; Stage 2 D{N} smoke disconfirm → continue Stage 2 from D{N+1} (or Stage 3 if D3 was the disconfirmed variant); Stage 3 smoke disconfirm → land M6 as the systematic-ablation appendix.

### 12.4 Stage 2 — Direction-quality variants (only if Stage 1 didn't terminate)

Each variant consumes a fresh direction artifact built by a new helper. Variants stack; the first to land ≤30% identifies the marginal load-bearing ingredient.

- [x] 12.4.1 **Implement `src/mechanistic/extract_activations.py --use-chat-template`** (~30 LOC): run extraction through `tokenizer.apply_chat_template(...)` matching `evaluate.py:143`, producing `refuse_activations_chat.pt` / `comply_activations_chat.pt`.
- [x] 12.4.2 **Implement `src/mechanistic/build_directions_v2.py`** (~150 LOC) with three composable flags: `--use-chat-template`, `--winsorize-pct 99.5` (clip activations element-wise per layer *before* the mean), `--orthogonalize-against-harmless-mean` (post-direction Gram-Schmidt against `mean_comply`).
- [x] 12.4.3 **Variant D1** (H2 isolation): build artifact with `--use-chat-template`. Re-run `abliterate.py` (bf16, vanilla projection, alpha=1.0) with the new artifact → `models/gemma-4-e4b-self-abliterated-d1/`. Smoke n=12 (6 should_refuse + 6 over-refuse). Apply three-band gate.
- [x] 12.4.4 **Variant D2** (D1 + H3): build artifact with `--use-chat-template --winsorize-pct 99.5`. Re-abliterate → `models/gemma-4-e4b-self-abliterated-d2/`. Smoke n=12. Apply three-band gate.
- [x] 12.4.5 **Variant D3** (D2 + H4): build artifact with `--use-chat-template --winsorize-pct 99.5 --orthogonalize-against-harmless-mean`. This is the full TrevorJS direction-build recipe with vanilla projection. Re-abliterate → `models/gemma-4-e4b-self-abliterated-d3/`. Smoke n=12. Apply three-band gate.
- [x] 12.4.6 For each variant: stop at the first to land ≤30% and run Stage 1.5 confirmation on it before declaring a paper headline. Verification per variant: artifact contains 42 unit-norm vectors of dim 2560; for D1 confirm chat-template-derived direction has cosine < 1.0 vs M2b's same-layer direction; for D2 confirm pre-clip max-norm exceeds post-clip max-norm by ≥20% at the peak refusal layer (L15 per M2b); for D3 confirm `|dot(direction, mean_comply_clipped)| < 1e-4` in float32 (D3 stacks on D2, so the orthogonalization target is the winsorized harmless mean, not raw `mean_comply`).
- [x] 12.4.6a If Stage 2 escalates to Stage 3 (no variant lands ≤30%), the cascade SHALL pass D3's direction artifact (the strongest direction-quality variant produced) into Stage 3a's biprojection — NOT M2b's raw-prompt artifact. This preserves any partial-effect signal accumulated in Stage 2 and isolates norm preservation as the marginal ingredient on top of D3.
- [x] 12.4.7 **Stage 2.5 (optional unstacked isolation):** only if a stacked variant cracks AND the operator wants single-variable causal claims. For whichever ingredient first crossed the gate, build the unstacked version (e.g., if D2 cracks, build "winsorize-only without chat-template" and test). ~30 min per variant. Skip if "ingredient X added on top of prior ingredients was the threshold" is acceptable as a "necessary in combination" claim.

### 12.5 Stage 3 — Norm-preserving biprojection (only if Stage 2 didn't terminate)

- [x] 12.5.1 **Stage 3a (local biprojection on D3 directions):** add a `--norm-preserving` flag to `src/abliterate/abliterate.py` (~40 LOC) implementing the magnitude+direction decomposition (preserving `‖W_i‖` per row). Unit-test: for random `d` and `W`, the resulting `W'` rows satisfy `‖W'_i‖ ≈ ‖W_i‖` to float32 precision. Run with D3's direction artifact. Smoke n=12. ~1 h impl + 30 min eval.
- [x] 12.5.2 **Stage 3b (faithful TrevorJS reproduction, only if 3a is ambiguous or operator-authorized):**
  - Pre-flight (mirrors M3 7.2): confirm the upstream repo's license is Apache 2.0 or compatible with Gemma weight redistribution; `df -h` confirms ≥10 GB free for any intermediate bf16 checkpoint.
  - Clone `https://github.com/TrevorS/gemma-4-abliteration` to `~/src/`, install deps in shared `.venv` (or report blockers in `docs/issues/`).
  - Run their script against `google/gemma-4-E4B-it`.
  - Run M3's `compute_diff.py` between our reproduction and the published TrevorJS weights — expect same 84-tensor footprint, σ₁/σ₂ in 50–200 range, |cos| > 0.5 between top-1 left singular vectors.
  - Smoke n=12 on the reproduction.
- [x] 12.5.3 Decision: if 3a reproduces ≈0%, paper claim is "biprojection is necessary on Gemma 4 because RMSNorm is sensitive to row-norm changes." If 3a fails but 3b succeeds, the gap was implementation-quality in 3a. If 3b also fails, the failure is environmental (tokenizer / chat template / generation settings); file an issue note and route to the negative-finding appendix branch.
- [x] 12.5.4 If either 3a or 3b lands ≤30% on the 12-prompt targeted smoke, route the winning checkpoint through Stage 1.5 (n=42 `should_refuse` confirmation) BEFORE Stage 4 — the n=12 smoke is too coarse for a paper headline, same as Stage 0a/Stage 2 winners.

### 12.6 Stage 4 — Full 344-prompt benchmark (only on the winner)

**HUMAN-IN-LOOP CHECKPOINT.** Confirm the headline number from Stage 1.5 is paper-grade with the operator before launching.

- [x] 12.6.1 Operator confirms Stage 1.5 result is paper-grade.
- [x] 12.6.2 Run the winning variant against all 344 benchmark prompts via the transformers backend (bf16) → `$RESULTS_DIR/stage4_<winner_slug>/evaluation_results.csv`.
- [x] 12.6.3 GGUF-convert the winning bf16 checkpoint (via `convert_hf_to_gguf.py`) and re-run the full benchmark via the llama.cpp backend → `$RESULTS_DIR/stage4_<winner_slug>_gguf/evaluation_results.csv`. This directly tests the "bf16-edited then GGUF-quantized still uncensored" prediction implied by HauhauCS's quantized success. Every Stage 0a / D1 / D2 / D3 / 3a / 3b winner is a bf16 safetensors checkpoint, so this step is unconditional.
- [x] 12.6.4 Compare per-category refusal rates against all rows in `STATUS_FOR_HUMAN.md` section (b); add a new row.
- [x] 12.6.5 Update the PAPER-HEADLINE-NUMBERS block in `STATUS_FOR_HUMAN.md` section (h).

### 12.7 Stage gating + cross-cutting checks

- [x] 12.7.1 After every stage, append a one-paragraph status to `STATUS_FOR_HUMAN.md` under a new `## M6 — Rank-1 Follow-up` section.
- [x] 12.7.2 Each stage SHALL terminate if wallclock exceeds 1.5× its budget; downgrade rather than running indefinitely.
- [x] 12.7.3 GPU lock acquired via `scripts/gpu_lock.sh` for any stage that runs inference; long evals use the `nohup`+`flock` pattern to survive subagent runtime cap.
- [x] 12.7.4 All canonical artifacts under `$RESULTS_DIR/`; in-repo handoff is short Markdown summaries only (raw CSVs/JSONs blocked by `.gitignore`).
- [x] 12.7.5 Default model routing: Sonnet 4.6 for Stage 0a/0b/D1/D2/D3 builds + evals + Stage 4; Opus 4.7 for the pre-launch component-target verification, gate decisions, framing assertions in commit messages, Stage 3 design (norm-preserving biprojection impl + TrevorJS reimpl + diff comparison), and any paper-claim drafting. The chat-template re-extraction is on the boundary — Sonnet for impl, Opus for design review of the resulting direction artifact.

### 12.8 Paper-side handoff (M5)

- [x] 12.8.1 The paper-side rephrasing of the headline claim and the new ablation table is M5's responsibility, not M6's. M6 hands off numbers; M5 writes prose.
- [x] 12.8.2 If a stage crack lands a positive result, M5 Section 7 (combined Abliteration & Weight-Diff) and Section 8 (M6 Rank-1 Cascade) reframe from "rank-1 fails on Gemma 4" to a causally-isolated single-ingredient finding per the decision tree in `docs/M6_PROPOSAL_RANK1_FOLLOWUP.md` Section 6. *(Section numbering under the 10-section outline post-Decision-7; the original task said "Section 7 (or new Section 7.5)" against the old 9-section outline.)*
- [x] 12.8.3 If no stage cracks, M6 lands as a systematic-ablation appendix to the existing M2c/M3 negative-finding paper.
