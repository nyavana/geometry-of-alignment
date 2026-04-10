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
3. **Why is abliteration so effective?** What makes a single rank-1 weight
   perturbation enough to remove refusal behavior?
4. **Can over-refusal be removed selectively?** Is it possible to keep refusal
   on harmful queries while eliminating refusal on benign-but-flagged ones?
5. **What do "uncensored" model releases actually change?** How do published
   uncensored variants differ from their original checkpoints at the
   weight-diff level?

## Models Studied

| Model | Architecture | Use |
|---|---|---|
| **Gemma 4 E4B-it** | 42-layer dense transformer (8B params) | GPU experiments via 8-bit quantization |
| **Qwen3.5-35B-A3B** | MoE, 256 experts | CPU-only weight-diff analysis (censored vs uncensored) |

## Repository Layout

```
.
├── src/
│   ├── benchmark/       # Refusal evaluation pipeline (llama.cpp + transformers)
│   ├── mechanistic/     # Activation extraction, refusal-direction computation, UMAP/t-SNE
│   ├── abliterate/      # Rank-1 weight perturbation, alpha sweeps, selective safety
│   └── weight_diff/     # CPU weight diff + SVD + MoE expert analysis
├── data/
│   └── benchmark_prompts.json   # 300–400 prompts across 8 categories
├── results/             # Activations, refusal directions, ablation results, figures
├── docs/                # Project plan, proposal, naming history
├── paper/               # Final paper drafts
└── requirements.txt
```

## Setup

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
```

Hardware used during development:

- **GPU**: NVIDIA RTX 4070 Ti Super 16 GB — Gemma 4 E4B at 8-bit (~7.5 GB VRAM)
- **CPU/RAM**: 100 GB DDR4 — Qwen3.5-35B safetensors weight-diff analysis

All Gemma experiments require `--use-8bit` to fit in VRAM.

## Running the Modules

Each module is invoked from the project root.

### Benchmark Evaluation

```bash
# llama.cpp backend (GGUF models)
python -m src.benchmark.evaluate \
    --backend llamacpp \
    --model <gguf_path> \
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
# Compare original vs uncensored checkpoints
python -m src.weight_diff.compute_diff \
    --original models/qwen-original/ \
    --modified models/qwen-uncensored/ \
    --output results/weight_diffs/qwen/

# SVD rank analysis of the diff
python -m src.weight_diff.svd_analysis \
    --results results/weight_diffs/qwen/weight_diff_results.json

# Which MoE experts were modified?
python -m src.weight_diff.moe_expert_analysis \
    --results results/weight_diffs/qwen/weight_diff_results.json
```

## Status

Work in progress as part of the EECS 6699 final project. See `docs/project_plan.md`
for the full plan and `docs/project_proposal.md` for the proposal.

## License

Academic / research use. License TBD.

## Acknowledgements

Course staff and peers from EECS 6699 (Spring 2026), Columbia University.
