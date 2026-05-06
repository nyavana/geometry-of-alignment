## Why

This project investigates where safety/refusal behavior lives in the weights of Gemma 4 E4B-it and characterizes why standard rank-1 abliteration (Arditi 2024) fails to remove it cleanly. Recent (2025–2026) literature documents two architectural quirks that break the textbook recipe on Gemma 4: four RMSNorm layers per decoder block (instead of two) and shared K/V tensors across layers 24–41. Multiple published Gemma 4 E4B uncensored variants — OBLITERATUS (whitened SVD + attention-head surgery + winsorized activations), TrevorJS (norm-preserving biprojection), and HauhauCS (aggressive GGUF) — each handle these quirks differently. The project's research questions: (1) where in the residual stream does refusal live? (2) does standard abliteration succeed on Gemma 4, and if not, what is the failure mode? (3) is selective de-alignment feasible — can over-refusal on emergency-medical queries be removed while refusal on harmful queries is preserved? (4) do the published variants converge on the same weight-space modification, or each find a different solution?

Running the entire pipeline — benchmark, mechanistic, abliteration, comparative weight diff — on a single model family makes the M2 → M3 cross-reference quantitative on the same parameter space: M2b's per-layer refusal directions can be cosine-compared against M3's top-1 left singular vectors of the published variants' weight diffs. That cross-reference is the central novel measurement of the project.

This change consolidates and supersedes the archived predecessors `alignment-geometry-study` and `autonomous-agent-pivot` (see `openspec/archive/`); together they are the single source of truth for milestones M0–M5.

## What Changes

The project's live deliverables, by milestone (full task list in `tasks.md`):

- **M2a — Benchmark evaluation.** Refusal-rate evaluation across the model lineup using `src/benchmark/`: `google/gemma-4-E4B-it` (base, GGUF + transformers), `google/gemma-4-E2B-it` (BF16 validation), `OBLITERATUS/gemma-4-E4B-it-OBLITERATED`, `TrevorJS/gemma-4-E4B-it-uncensored`, `HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive`, plus the project's own M2c-abliterated variants. Phrasing- and context-sensitivity sweeps; refusal heatmap.
- **M2b — Mechanistic analysis.** Activation extraction on Gemma 4 E4B 8-bit, per-layer refusal-direction computation, layer signal-strength + sliding/global comparison, PCA rank analysis, UMAP/t-SNE visualizations, cross-precision validation against E2B BF16.
- **M2c — Abliteration + selective safety.** This project's own rank-1 abliteration of Gemma 4 E4B; alpha sweep, layer-subset sweep, prompt-count sweep, random-direction control; capability preservation (MMLU + GSM8K subsets); category-specific refusal directions; selective abliteration removing emergency-medical refusal while preserving should-refuse refusal. Any RMSNorm / shared-K/V failure mode is documented as a project finding.
- **M3 — Comparative weight diff (CPU-only).** Element-wise diff + SVD rank analysis between base and (a) `OBLITERATUS/...` (primary) and (b) `TrevorJS/...` (secondary, with fallback to OBLITERATUS-only if shape/key pre-flight fails). Cross-method overlay on per-layer Frobenius and on top-1 singular vectors. Cross-reference of singular vectors against M2b refusal directions. Architectural-quirk handling: shared K/V tensors are de-duplicated so per-layer plots count each unique tensor once.
- **M4 — Human verification gate.** `STATUS_FOR_HUMAN.md` summarizes all M2/M3 results with CSV/figure citations and the green-light sentence the operator writes to authorize M5.
- **M5 — Paper + slides.** Nine-section paper. Section 7 is "Comparative weight diff across published Gemma 4 E4B abliterations" (no MoE / router / expert content). Sources file maps every numeric claim to a file path + commit hash.

Predecessor scenarios from `autonomous-agent-pivot/research-paper` (writeup gated by human verification, all numeric claims traceable) are merged inline into this change's `research-paper` spec. The project does not include MoE / router / expert analysis.

## Capabilities

This change owns six capabilities, all defined as spec deltas under `specs/`:

- `weight-diff-analysis` — comparative cross-method weight diff (OBLITERATUS, TrevorJS); SVD rank analysis; cross-reference singular vectors against M2b refusal directions; architectural-quirk handling for Gemma 4 shared K/V.
- `benchmark-evaluation` — multi-backend refusal-rate evaluation across the model lineup (upstream llama.cpp `llama-server` over HTTP for GGUF; `transformers` `--use-8bit` for HF safetensors on GPU); phrasing- and context-sensitivity sweeps.
- `activation-analysis` — refusal-direction extraction via mean-diff per layer; signal-strength + PCA rank analysis; UMAP/t-SNE visualization; cross-precision validation.
- `abliteration-engine` — rank-1 weight perturbation `W' = W - α · d · (d^T W)`; alpha / layer-subset / prompt-count sweeps; random-direction control; capability preservation; selective safety via category-specific directions.
- `autonomous-execution` — six-worktree dispatch contract with GPU-lock policy, branch-scoped `RESULTS_DIR`, commit-and-push protocol, stop-at-section-boundary contract.
- `research-paper` — nine-section paper; Section 7 framed as comparative weight diff; writeup gated by `STATUS_FOR_HUMAN.md` green-light; numeric claims traceable to source artifacts.

## Impact

- **Compute**: GPU (NVIDIA 4070 Ti Super, 16 GB) for M2 (Gemma 4 E4B 8-bit, ~7.5 GB VRAM); CPU + ~46 GB physical RAM (WSL2-capped) + 12 GB swap for M3 (~17 GB-per-variant Gemma safetensors weight diff; peaks ~32–35 GB per run). M3 runs concurrently with M2 — no GPU contention. Variant diffs run sequentially, not in parallel, since two simultaneous runs (~68 GB combined) exceed available RAM.
- **Models**: `google/gemma-4-E4B-it` (base), `google/gemma-4-E2B-it` (validation), `OBLITERATUS/gemma-4-E4B-it-OBLITERATED` (safetensors + GGUF), `TrevorJS/gemma-4-E4B-it-uncensored` (bf16 safetensors), `HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive` (GGUF). No MoE models.
- **Dependencies**: `safetensors`, `torch`, `transformers`, `bitsandbytes` in `requirements.txt`; `gguf` pinned to llama.cpp's git tree because pypi releases lag (the upstream `convert_hf_to_gguf.py` references symbols like `MODEL_ARCH.GEMMA4` that aren't in pypi yet). `llama-cpp-python` is removed; the GGUF backend now talks HTTP to upstream llama.cpp's `llama-server`, built from source into `shared/llama.cpp-cuda/` against `apt install nvidia-cuda-toolkit` (~2.5 GB, host-only) and surfaced through `shared/env.sh`.
- **Disk**: ~25 GB for base + E2B (existing) + ~34 GB for OBLITERATUS + TrevorJS safetensors + ~5 GB for HauhauCS GGUF. Pre-flight `df -h` check in M3 step 1 verifies budget.
- **Authorization**: instructor's autonomous-agent authorization (recorded in auto-memory `project_authorization.md`) is the standing license for agents to dispatch and merge without per-step human approval.
- **Risk mitigation**: the `weight-diff-analysis` `Architectural quirk handling` requirement makes the shared-K/V double-counting risk an explicit acceptance criterion; the TrevorJS-fallback rule degrades M3 to N=1 (OBLITERATUS only) without halting the project.
- **Provenance**: predecessors `alignment-geometry-study` and `autonomous-agent-pivot` are archived under `openspec/archive/<name>/` with `_NOTE.md` redirects; `src/weight_diff/moe_expert_analysis.py` is no longer in the codebase.
