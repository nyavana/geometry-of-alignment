## Why

LLM safety alignment can over-refuse beneficial queries -- an on-device model asked for emergency Heimlich maneuver instructions refuses, potentially costing lives in remote scenarios without connectivity. Meanwhile, "cracked" open-source models (abliterated Gemma 4, uncensored Qwen3.5) remove safety with a single-direction weight edit, revealing that alignment is shockingly low-dimensional. This project experimentally investigates where safety lives in modern LLM weights, why it breaks so easily, and whether selective de-alignment (remove over-refusal while preserving genuine safety) is possible. The work serves as the final research project for EECS 6699: Mathematics of Deep Learning at Columbia University.

## What Changes

- Build a categorized benchmark of 300-400 safety-adjacent prompts (emergency medical, survival, gray-zone, genuinely harmful, safe controls) and evaluate refusal rates across state-of-the-art models (Gemma 4 E4B, Qwen3.5-35B-A3B) and their cracked counterparts
- Extract layer-by-layer activations from Gemma 4 E4B-it to locate the refusal direction in residual stream space, analyze its dimensionality via PCA, and map it across the 42-layer sliding/global attention architecture
- Reproduce abliteration from scratch on Gemma 4 E4B, run ablation sweeps (alpha, layer subsets, prompt count, random control), and attempt category-specific selective safety
- Reverse-engineer published cracked models (Qwen3.5-35B-A3B-Uncensored) by computing weight diffs, SVD rank analysis, and MoE expert-level modification mapping
- Produce a research paper with survey (RLHF, DPO, representation engineering, abliteration literature) and experimental sections, plus presentation slides

## Capabilities

### New Capabilities
- `benchmark-evaluation`: Categorized over-refusal and safety benchmark with automated refusal classification and cross-model evaluation pipeline using llama.cpp
- `activation-analysis`: Layer-wise residual stream activation extraction, refusal direction computation, PCA rank analysis, and UMAP/t-SNE visualization for Gemma 4 E4B
- `abliteration-engine`: Abliteration implementation with ablation study framework (alpha sweep, layer subset sweep, prompt count sweep, random control) and selective safety via category-specific refusal directions
- `weight-diff-analysis`: CPU-based weight diff computation between original and cracked model pairs, SVD rank analysis, per-layer Frobenius norm mapping, and MoE expert modification heatmaps
- `research-paper`: Integrated LaTeX paper with survey sections (alignment techniques, representation engineering, abliteration), experimental results, and course connections (over-parametrization, matrix perturbation, NTK)

### Modified Capabilities

## Impact

- **Compute**: Gemma 4 E4B-it at 8-bit (~7.5 GB VRAM) on 4070 Ti Super 16GB for activation/abliteration work; Qwen weight diffs on CPU using 100GB RAM; llama.cpp for benchmark inference
- **Models**: google/gemma-4-E4B-it, google/gemma-4-E2B-it, Qwen/Qwen3.5-35B-A3B, HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive
- **Dependencies**: PyTorch, transformers, bitsandbytes, llama-cpp-python, scikit-learn, safetensors, matplotlib/seaborn/plotly, umap-learn
- **Timeline**: ~4 weeks with 4 team members working in parallel
- **Ethics**: All work framed as understanding alignment to improve it; selective safety is the constructive application; paper includes ethics discussion
