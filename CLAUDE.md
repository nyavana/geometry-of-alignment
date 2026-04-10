# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project for EECS 6699 (Mathematics of Deep Learning, Columbia University) investigating where safety/refusal behavior lives in LLM weights, why abliteration (a rank-1 weight perturbation) can remove it, and whether selective de-alignment is possible (remove over-refusal on medical queries while keeping refusal on harmful ones).

Primary models: Gemma 4 E4B-it (42 layers, dense, GPU via 8-bit quantization) and Qwen3.5-35B-A3B (MoE, CPU-only weight diff analysis).

## Running Modules

All modules run as Python module invocations from the project root. There is no build system, test suite, or linter configured.

```bash
# Install dependencies
pip install -r requirements.txt

# Benchmark evaluation (llama.cpp backend)
python -m src.benchmark.evaluate --backend llamacpp --model <gguf_path> --benchmark data/benchmark_prompts.json --output results/<model_name>/

# Benchmark evaluation (transformers backend, for abliterated models)
python -m src.benchmark.evaluate --backend transformers --model google/gemma-4-E4B-it --benchmark data/benchmark_prompts.json --output results/<model_name>/ --use-8bit

# Analyze results across models
python -m src.benchmark.analyze_results --results-dir results/

# Extract activations and compute refusal directions
python -m src.mechanistic.extract_activations --model google/gemma-4-E4B-it --benchmark data/benchmark_prompts.json --use-8bit --output results/activations/

# Layer analysis (requires activations extracted first)
python -m src.mechanistic.layer_analysis --activations results/activations/

# Visualization (UMAP/t-SNE)
python -m src.mechanistic.visualize --activations results/activations/ --method umap

# Abliterate a model (requires refusal directions computed first)
python -m src.abliterate.abliterate --model google/gemma-4-E4B-it --directions results/activations/refusal_directions.pt --alpha 1.0 --use-8bit --output models/gemma-4-e4b-abliterated/

# Ablation study (alpha sweep, layer sweep, random control)
python -m src.abliterate.ablation_study --model google/gemma-4-E4B-it --activations results/activations/ --benchmark data/benchmark_prompts.json --use-8bit

# Selective safety experiments
python -m src.abliterate.selective_safety --model google/gemma-4-E4B-it --benchmark data/benchmark_prompts.json --use-8bit

# Weight diff (CPU-only, needs both model directories with safetensors)
python -m src.weight_diff.compute_diff --original models/qwen-original/ --modified models/qwen-uncensored/ --output results/weight_diffs/qwen/

# SVD analysis of weight diffs
python -m src.weight_diff.svd_analysis --results results/weight_diffs/qwen/weight_diff_results.json

# MoE expert analysis
python -m src.weight_diff.moe_expert_analysis --results results/weight_diffs/qwen/weight_diff_results.json
```

## Architecture

Four independent modules in `src/`, each runnable as a standalone pipeline:

- **`src/benchmark/`** — Refusal evaluation pipeline. `evaluate.py` runs models (llama.cpp or transformers) on prompts from `data/benchmark_prompts.json`. `classify_refusal.py` uses regex-based two-stage classification (refusal patterns vs compliance patterns). `analyze_results.py` generates heatmaps and comparison charts.

- **`src/mechanistic/`** — Activation extraction and analysis. `extract_activations.py` hooks into transformer layers via `register_forward_hook`, collects residual stream outputs, and computes refusal directions (mean_refuse - mean_comply, normalized). `layer_analysis.py` runs PCA rank analysis and signal strength per layer. `visualize.py` produces UMAP/t-SNE projections. Key class: `ActivationCollector` manages hooks and stores activations on CPU.

- **`src/abliterate/`** — Abliteration implementation. `abliterate.py` applies `W_new = W - alpha * d * (d^T @ W)` to `o_proj.weight` and `down_proj.weight` via `_project_out()`. `ablation_study.py` sweeps alpha (0-2.0), layer subsets (global-only, sliding-only, etc.), and includes random direction control. `selective_safety.py` computes category-specific refusal directions and tests removing medical over-refusal while keeping harmful-query refusal.

- **`src/weight_diff/`** — CPU-based weight comparison. `compute_diff.py` loads safetensors, computes element-wise diffs with SVD rank analysis. `svd_analysis.py` visualizes modification ranks and per-layer changes. `moe_expert_analysis.py` maps which MoE experts/routers/shared-experts were modified in cracked Qwen models.

### Cross-module dependencies

`ablation_study.py` imports from both `mechanistic.extract_activations` (for `load_model_and_tokenizer`) and `benchmark.evaluate` (for `evaluate_with_transformers`). `selective_safety.py` imports from `mechanistic.extract_activations` and `abliterate.ablation_study`. The `weight_diff` module is fully independent.

## Key Data

- `data/benchmark_prompts.json` — 300-400 prompts across 8 categories with `id`, `category`, `expected` (refuse/comply), `prompt`, and `variants` fields
- `results/activations/` — `.pt` files: `refuse_activations.pt`, `comply_activations.pt`, `refusal_directions.pt` (dict keyed by layer index)
- `results/ablation_results/` — JSON sweep results
- `results/weight_diffs/` — JSON diff results and SVD `.pt` files
- `results/figures/` — Generated plots (PNG)

## Gemma 4 E4B Architecture Constants

Used throughout the codebase — verify against model config if models are updated:
- 42 layers: 35 sliding attention + 7 global attention
- Global attention layer indices: `[5, 11, 17, 23, 29, 35, 41]`
- Hidden size: 2560
- At 8-bit: ~7.5 GB VRAM on a 16 GB 4070 Ti Super

## Hardware Constraints

- GPU: NVIDIA 4070 Ti Super 16GB — used for Gemma 4 E4B (8-bit quantization required)
- CPU/RAM: 100GB DDR4 — used for Qwen3.5-35B weight diff analysis via safetensors
- Always pass `--use-8bit` for GPU-based Gemma experiments to stay within VRAM budget
