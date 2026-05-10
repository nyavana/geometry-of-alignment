# Geometry of Alignment

> How safety lives and dies in modern LLMs.

A research project for **EECS 6699 (Mathematics of Deep Learning)** at
Columbia University, investigating the mechanistic basis of safety alignment
in large language models: where refusal behavior is encoded, why it is so
fragile that a single rank-1 weight perturbation can eliminate it, and whether
selective de-alignment (removing over-refusal while keeping genuine safety) is
possible.

## Research Questions

1. **How prevalent is over-refusal in practice?** When do aligned models
   refuse beneficial queries (e.g. emergency medical advice)?
2. **Where does refusal live?** Which layers and subspaces of the residual
   stream encode the refusal direction, and what dimensionality does it
   occupy?
3. **Is rank-1 abliteration universally effective, and if not, why?** Recent
   work reports the standard mean-diff recipe fails on certain post-2024
   architectures (Gemma 4 in particular). What does that look like
   empirically, and what does it tell us about the geometry of refusal?
4. **Can over-refusal be removed selectively?** Is it possible to keep refusal
   on harmful queries while eliminating refusal on benign-but-flagged ones?
5. **What do "uncensored" model releases actually change?** How do published
   uncensored variants differ from their original checkpoints at the
   weight-diff level, and do they share a common modification subspace?

## Models Studied

| Model | Architecture | Use |
|---|---|---|
| **Gemma 4 E4B-it** (base) | 42-layer dense transformer (~4B effective) | GPU experiments via 8-bit quantization |
| **Gemma 4 E2B-it** (validation) | dense, smaller | Cross-precision validation |
| **OBLITERATUS / Gemma-4-E4B-it-OBLITERATED** | published abliteration (whitened SVD + attention head surgery + winsorized activations; 21 of 42 layers modified) | CPU weight-diff target + behavioral eval |
| **TrevorJS / Gemma-4-E4B-it-uncensored** | published abliteration (norm-preserving biprojection) | CPU weight-diff target + behavioral eval |
| **HauhauCS / Gemma-4-E4B-Uncensored-HauhauCS-Aggressive** | published abliteration (GGUF only) | Behavioral eval only |

## Repository Layout

```
.
├── src/
│   ├── benchmark/       # Refusal evaluation pipeline (llama.cpp + transformers)
│   ├── mechanistic/     # Activation extraction, refusal-direction computation, UMAP/t-SNE
│   ├── abliterate/      # Rank-1 weight perturbation, alpha sweeps, selective safety
│   └── weight_diff/     # CPU weight diff + SVD + cross-variant comparison
├── data/
│   └── benchmark_prompts.json   # 300–400 prompts across 8 categories
├── results/             # Activations, refusal directions, ablation results, figures
├── docs/                # Project plan, proposal, naming history
├── paper/               # Final paper drafts
├── model -> ../shared/model      # symlink, excluded via .git/info/exclude
├── .venv -> ../shared/.venv      # symlink, excluded via .git/info/exclude
└── requirements.txt
```

Development happens across multiple sibling git worktrees (`gb-ablit`, `gb-bench`, `gb-env`, `gb-mech`, `gb-paper`, `gb-wdiff`) that share a non-versioned `/home/nyavana/columbia/6699/shared/` sidecar holding the model checkpoints (25 GB), Python venv, Hugging Face cache, and per-branch results. See `CLAUDE.md` (Worktrees & Shared Sidecar) for the full diagram.

## Setup

```bash
# Activate the shared Python 3.12 environment from any worktree
source /home/nyavana/columbia/6699/shared/env.sh   # sets HF_HOME, RESULTS_DIR, etc.
source .venv/bin/activate                          # symlink resolves to shared/.venv
```

To bootstrap the shared venv from scratch:

```bash
python3.12 -m venv /home/nyavana/columbia/6699/shared/.venv
source /home/nyavana/columbia/6699/shared/.venv/bin/activate
pip install -r requirements.txt
```

The GGUF inference path uses upstream llama.cpp's `llama-server`, built from source into `/home/nyavana/columbia/6699/shared/llama.cpp-cuda/` (CUDA via `apt install nvidia-cuda-toolkit`, sm_89 only). `shared/env.sh` prepends its `bin/` to `PATH` and exports `LD_LIBRARY_PATH` for the bundled `.so`s — so once `shared/env.sh` is sourced, `llama-server` resolves to the CUDA build. To rebuild from a newer release: `git clone https://github.com/ggml-org/llama.cpp ~/src/llama.cpp && cd ~/src/llama.cpp && cmake -B build -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build build -j$(nproc) && cmake --install build --prefix /home/nyavana/columbia/6699/shared/llama.cpp-cuda`.

Launch with the GGUF you want to evaluate:

```bash
llama-server -m path/to/model.gguf -ngl 99 --host 127.0.0.1 --port 8088
```

(Use port 8088, not 8080 — Windows-side WSL2 already binds 8080 on this host.)

`evaluate.py --backend llamacpp` then sends OpenAI-compatible chat completions to that endpoint; `--server-url` defaults to `http://127.0.0.1:8088`.

Hardware used during development:

- **GPU**: NVIDIA RTX 4070 Ti Super 16 GB — Gemma 4 E4B at 8-bit (~7.5 GB VRAM)
- **CPU/RAM**: ~46 GB physical + 12 GB swap (WSL2-capped) — Comparative safetensors weight-diff across published Gemma 4 E4B uncensored variants (~17 GB per variant; ~32–35 GB peak per run, so variants are diffed sequentially, not in parallel)

All Gemma experiments require `--use-8bit` to fit in VRAM.

## Running the Modules

Each module is invoked from the project root.

### Benchmark Evaluation

```bash
# llama.cpp backend (GGUF models — start llama-server first; see Setup above)
python -m src.benchmark.evaluate \
    --backend llamacpp \
    --model <label> \
    --server-url http://127.0.0.1:8088 \
    --benchmark data/benchmark_prompts.json \
    --output results/<model_name>/

# transformers backend (HF models, abliterated variants)
python -m src.benchmark.evaluate \
    --backend transformers \
    --model google/gemma-4-E4B-it \
    --benchmark data/benchmark_prompts.json \
    --output results/<model_name>/ \
    --use-8bit

# Cross-model comparison
python -m src.benchmark.analyze_results --results-dir results/
```

### Mechanistic Analysis

```bash
# Extract residual-stream activations and compute refusal directions
python -m src.mechanistic.extract_activations \
    --model google/gemma-4-E4B-it \
    --benchmark data/benchmark_prompts.json \
    --use-8bit \
    --output results/activations/

# Per-layer rank and signal-strength analysis
python -m src.mechanistic.layer_analysis --activations results/activations/

# UMAP / t-SNE visualization
python -m src.mechanistic.visualize \
    --activations results/activations/ \
    --method umap
```

### Abliteration

```bash
# Apply rank-1 perturbation to o_proj and down_proj
python -m src.abliterate.abliterate \
    --model google/gemma-4-E4B-it \
    --directions results/activations/refusal_directions.pt \
    --alpha 1.0 \
    --use-8bit \
    --output models/gemma-4-e4b-abliterated/

# Alpha sweep, layer subset sweep, random direction control
python -m src.abliterate.ablation_study \
    --model google/gemma-4-E4B-it \
    --activations results/activations/ \
    --benchmark data/benchmark_prompts.json \
    --use-8bit

# Selective safety experiments (medical over-refusal vs harmful refusal)
python -m src.abliterate.selective_safety \
    --model google/gemma-4-E4B-it \
    --benchmark data/benchmark_prompts.json \
    --use-8bit
```

### Weight Diff Analysis (CPU)

```bash
# Compare base Gemma 4 E4B-it against the OBLITERATUS abliteration
python -m src.weight_diff.compute_diff \
    --original model/gemma-4-E4B-it/ \
    --modified model/OBLITERATUS-gemma-4-E4B-it-OBLITERATED/ \
    --output results/weight_diffs/gemma_obliteratus/

# Compare base against TrevorJS norm-preserving biprojection variant
python -m src.weight_diff.compute_diff \
    --original model/gemma-4-E4B-it/ \
    --modified model/TrevorJS-gemma-4-E4B-it-uncensored/ \
    --output results/weight_diffs/gemma_trevorjs/

# SVD rank analysis of each diff (run per variant)
python -m src.weight_diff.svd_analysis \
    --results results/weight_diffs/gemma_obliteratus/weight_diff_results.json
```

## Status

Experimental work for the EECS 6699 final project is complete (M0–M4 + M6);
paper write-up (M5) is in progress (section 8 drafted; sections 1, 4, 5, 6,
7, 9 pending). See `STATUS_FOR_HUMAN.md` at the repo root for the
operator-review status doc with all headline numbers, figure paths, and
verification checklist; `docs/findings/M6_PROPOSAL_RANK1_FOLLOWUP.md` for the M6
cascade design + per-stage results; `docs/planning/project_plan.md` for the original
execution plan; `docs/planning/project_proposal.md` for the original proposal.

### Headline findings

- **Refusal is rank-1 in activations, but not in weights.** M2b shows that
  the activation-derived refusal direction is effectively rank-1 (top-1 PC
  captures 86.6% of |Δμ|² at the peak band L4–L17, peak at L15 with Cohen's
  d 2.87). M3 shows the *weight* diffs that produce uncensored behavior are
  not: OBLITERATUS edits are multi-rank (median rank_95 = 6) and TrevorJS
  edits are pure rank-1 norm-preserving biprojections — the two methods
  modify nearly orthogonal subspaces (median |cos| = 0.08) and neither's
  top-1 singular vector aligns with M2b's activation direction (median
  |cos| ≈ 0.04).
- **Standard rank-1 mean-diff abliteration with the Mlabonne/Arditi recipe
  is empirically ineffective on Gemma 4 E4B**, regardless of edit-time
  precision. The M2c sweep (α∈[0, 2.0], 9 layer subsets, random control)
  leaves the int8 self-abliterated model's `should_refuse` rate at 100%;
  M6 Stage 0a confirms the same 100% under bf16 in-place edit, ruling out
  the bnb int8 path as the cause.
- **A single direction-quality ingredient is load-bearing on Gemma 4: two-
  pass Gram-Schmidt orthogonalization of the refusal direction against the
  harmless mean.** M6 isolates this via a five-stage causal cascade. With
  this step (D3 variant: chat-template-applied activations + per-layer
  winsorization at 99.5% + Gram-Schmidt vs harmless mean), vanilla rank-1
  projection at α=1.0 cuts `should_refuse` from 100% to 40.5% at n=42 — a
  60% relative reduction. Without it, no variant tested affects
  `should_refuse` at all. Norm-preserving biprojection adds nothing on top
  (H5 refuted): vanilla projection only changes per-row L2 norms by
  0.03–0.07% on average (max 2.84%), so there is essentially no row-norm
  change for biprojection to preserve.
- **Refusal on Gemma 4 is not cleanly a rank-1 phenomenon.** The persistent
  ~40-50% residual after the cleanest rank-1 recipe concentrates on the
  most extreme topics (CSAM, ICS/hospital malware, weapons), suggesting a
  strong core safety circuit that single-direction abliteration cannot
  reach. This corroborates M3's observation that OBLITERATUS uses median
  rank_95 = 6 to fully uncensor the same base model. Full removal needs
  multi-rank descent (out of scope here; future work).
- **Selective safety geometry is clean.** Per-category refusal directions
  (emergency_medical, wilderness_survival, home_safety, chemistry_safety,
  mental_health) form a tight +0.93 pairwise cluster and are orthogonal to
  the `should_refuse` direction (mean cosine −0.015). The geometry permits
  selective abliteration even though the magnitude-side constraint above
  blocks it from working with the standard rank-1 recipe.
- **Over-refusal hypothesis only weakly confirmed at the base level.**
  Base Gemma 4 E4B GGUF refuses only 2% of `emergency_medical` prompts
  (vs 100% of `should_refuse`); E2B BF16 is slightly more conservative
  (12% / 95.2%). The "models refuse helpful medical questions" trope is
  not a strong feature of the Gemma 4 family on this benchmark.

See `STATUS_FOR_HUMAN.md` (b) for the full benchmark table and (h) for
paper-grade headline numbers; `docs/findings/M6_PROPOSAL_RANK1_FOLLOWUP.md` §7 and
`paper/sections/08_rank1_cascade.md` for the M6 cascade.

## License

Academic / research use. License TBD.

## Acknowledgements

Course staff and peers from EECS 6699 (Spring 2026), Columbia University.
